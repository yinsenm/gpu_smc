#include <iostream>
#include <cmath>
#include <cstdio>
#include <random>
#include <chrono>
#include <arrayfire.h>
#include <Eigen/Dense>

#define PI 3.141592653589793238463f

// clear default macros
#undef min
#undef max

// keep track of time
class Timer {
#ifdef __linux__
	// use when in linux
	std::chrono::high_resolution_clock::time_point tic, toc;
#elif _WIN32
	// used when in windows
	std::chrono::time_point<std::chrono::steady_clock> tic, toc;
#endif
	std::chrono::duration<float> duration;
	const char* name;
public:
	Timer(const char* pname) {
		tic = std::chrono::high_resolution_clock::now();
		name = pname;
	}

	float get_time_spent() {
		toc = std::chrono::high_resolution_clock::now();
		duration = toc - tic;
		return duration.count();
	}

	~Timer() {
		toc = std::chrono::high_resolution_clock::now();
		duration = toc - tic;
		std::cout << name << " took " << duration.count() << "s" << std::endl;
	}
};

class smc {
private:
	int seed;
	std::default_random_engine rng;
	float loglik;	// log-likelihood

	float q, r;	// innovation and observation noise
	float A, B;	// hyper-parameters for the priors
	float time;	// keep track of time

	int T;	// number of data points
	int numMCMC;
	int numParticle;
	bool verbose = true;
	bool debug = false;

	// particle system (numParticle x T)
	af::array x; // latent, numParticle x T
	af::array w; // weight, numParticle x T
	af::array a; // ancestor, numParticle x T

	// sampling system
	Eigen::MatrixXf p;	// (numMCMC x 3) q, r, loglik
	Eigen::MatrixXf X;	// (numMCMC x T)

	// gpu data
	af::array d_x, d_y;
public:
	smc(int pseed); // contructor
	void set_param(float pq, float pr);
	void set_prior(float pA, float pB);
	void set_mcmc(int pnumMCMC, int pnumParticle);
	af::array resample_metropolis(const af::array &weights, int iterations);
	float h(float xt);
	float f(float xt, float t);
	af::array h(const af::array &xt);
	af::array f(const af::array &xt, float t);
	void generate_data(float tq, float tr, int tT);
	float log_dinvgamma(float x, float shape, float scale);
	float pf(float cq, float cr);
	void pmmh(float qsize, float rsize);
};

smc::smc(int pseed) {
	seed = pseed;
	try {
		af::setDevice(0);
		af::info();
		af::setSeed(seed);		// set seed for arrayfire
		rng.seed(seed);
	}
	catch (af::exception& e) {
		std::cerr << e.what() << "\n";
		throw;
	}
}

// set parameter for simulation
void smc::set_param(float pq, float pr) {
	q = pq;
	r = pr;
	if (q <= 0 || r <= 0) {
		std::cerr << "q and r should be larger than 0" << "\n";
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		printf("------------------------------------------------\n");
		printf("The initial parameters for SMC are ...\n");
		printf("q: %8.4f, r: %8.4f, T: %04d\n", q, r, T);
		printf("------------------------------------------------\n\n");
	}
}

// set prior for smc
void smc::set_prior(float pA, float pB) {
	A = pA;
	B = pB;

	if (A <= 0 || B <= 0) {
		std::cerr << "A and B should be larger than 0" << "\n";
		exit(EXIT_FAILURE);
	}

	if (verbose) {
		printf("------------------------------------------------\n");
		printf("The prior on the SMC are ...\n");
		printf("A: %8.4f, B: %8.4f\n", A, B);
		printf("------------------------------------------------\n\n");
	}

}


void smc::set_mcmc(int pnumMCMC, int pnumParticle) {
	if (pnumMCMC <= 0) {
		std::cerr << "the number of MCMC samples should be larger than 0." << std::endl;
		exit(EXIT_FAILURE);
	}

	if (pnumParticle <= 0) {
		std::cerr << "the number of particle samples should be larger than 0." << std::endl;
		exit(EXIT_FAILURE);
	}

	numMCMC = pnumMCMC;
	numParticle = pnumParticle;

	if (verbose) {
		printf("------------------------------------------------\n");
		printf("The settings for SMC + MCMC are ...\n");
		printf("number of MCMC samples: %04d\nnumber of particles: %04d\n", numMCMC, numParticle);
		printf("------------------------------------------------\n\n");
	}

}

// log density of an inversed gamma distribution given the shape and rate parameters
float smc::log_dinvgamma(float x, float shape, float scale) {
	if (shape <= 0 || scale <= 0 || x <= 0) {
		std::cerr << "x, shape and scale should be larger than 0." << std::endl;
		exit(EXIT_FAILURE);
	}
	return shape * log(scale) - lgamma(shape) - (shape + 1) * log(x) - scale / x;
}

// generate data
void smc::generate_data(float tq, float tr, int tT) {
	T = tT;
	std::normal_distribution<float> rnorm(0.0f, 1.0f);
	float *h_x = new float[T];
	float *h_y = new float[T];
	h_x[0] = 0.0f; // initial condition
	for (int t = 0; t < T; t++) {
		if (t < T - 1) {
			h_x[t + 1] = f(h_x[t], (float)t) + std::sqrt(tq) * rnorm(rng);
		}
		h_y[t] = h(h_x[t]) + std::sqrt(tr) * rnorm(rng);
	}

	// set device data
	d_x = af::array(T, 1, h_x);
	d_y = af::array(T, 1, h_y);
	delete[] h_x;
	delete[] h_y;
}

// particle filter
float smc::pf(float cq, float cr) {
	af::timer::start();
	af::array cloglik = af::constant(0, 1, 1, f32);
	af::array max_log_weight = af::constant(0, 1, 1, f32);
	af::array ind = af::seq(numParticle);
	af::array xpred = af::constant(0, numParticle, 1, f32);
	af::array ypred = af::constant(0, numParticle, 1, f32);
	af::array weights = af::constant(0, numParticle, 1, f32);

	// clear particle filters
	x = af::constant(0, numParticle, T, f32);	// particles
	a = af::constant(0, numParticle, T, u32);	// ancestor indices
	w = af::constant(0, numParticle, 1, f32);	// weights

	// initial latent state
	x.col(0) = 0.0f;
	w = 1.0f / numParticle;
	for (int t = 0; t < T; t++) {
		if (t != 0) {
			ind = resample_metropolis(w, 2);
			xpred = f(x.col(t - 1), (float)(t - 1));	// prediction
			x.col(t) = xpred(ind) + std::sqrt(cq) * af::randn(numParticle, 1, f32); // mutation
			a.col(t) = ind; // store ancestor indices
		}

		// compute importance weights
		ypred = h(x.col(t));
		weights = -0.5f * std::log(2.0f * PI * cr) - 1.0f / (2.0f * cr) * af::pow(af::tile(d_y(t), numParticle) - ypred, 2.0);
		max_log_weight = af::max(weights);
		weights = af::exp(weights - af::tile(max_log_weight, numParticle));

		// compute loglikelihood
		cloglik += max_log_weight + af::log(af::sum(weights)) - std::log(numParticle);
		w = weights / af::tile(af::sum(weights), numParticle);
	}

	// backward pass
	// generate the trajectories from ancestor indices
	ind = a.col(af::end);
	for (int t = T - 2; t >= 0; t--) {
		x.col(t) = x(ind, t);
		ind = a(ind, t);
	}

	if (debug) {
		std::cout << "q: " << cq << " r: " << cr << std::endl;
		af_print(cloglik);
		printf("PF elpased seconds: %g\n", af::timer::stop());
	}
	return cloglik.scalar<float>();
}


// Metropolis Hasting 
void smc::pmmh(float qsize, float rsize) {
	float loglik_prop, q_prop, r_prop;
	float begin, end;  // keep trakck of time
	float acceptprob;
	float accept_counter = 0;
	bool accept;
	af::array J;
	std::normal_distribution<float> rnorm(0.0f, 1.0f);
	std::uniform_real_distribution<float> runif(0.0, 1.0);

	// initial parameter
	// clear sampling matrix
	X = Eigen::MatrixXf::Zero(T, numMCMC);
	p = Eigen::MatrixXf::Zero(3, numMCMC);

	// run one particle filter
	loglik = pf(q, r);
	J = af::where(af::tile(af::randu(1, 1, f64), numParticle) < af::accum(w.col(af::end)))(0);
	//af::sync();
	af::lookup(x, J, 0).host(X.col(0).data());
	p.col(0) << q, r, loglik;

	Timer timer("PMMH");
	begin = timer.get_time_spent();

	for (int k = 1; k < numMCMC; k++) {
		q_prop = q + qsize * rnorm(rng);
		r_prop = r + rsize * rnorm(rng);

		if (q_prop <= 0 || r_prop <= 0) {
			accept = false;
		}
		else {
			// run another particle filer with updated paramter
			loglik_prop = pf(q_prop, r_prop);

			// compute acceptance probaility
			acceptprob = loglik_prop - loglik;
			acceptprob += log_dinvgamma(q_prop, A, B) +
				log_dinvgamma(r_prop, A, B) -
				log_dinvgamma(q, A, B) -
				log_dinvgamma(r, A, B);
			accept = runif(rng) < std::exp(acceptprob);
		}

		if (accept) {
			q = q_prop;
			r = r_prop;
			loglik = loglik_prop;
			p.col(k) << q, r, loglik;

			// Draw J
			J = af::where(af::tile(af::randu(1, 1, f64), numParticle) < af::accum(w.col(af::end)))(0);
			af::lookup(x, J, 0).host(X.col(k).data());
			accept_counter++;
		}
		else {
			p.col(k) = p.col(k - 1);
			X.col(k) = X.col(k - 1);
		}

		// print diagnostic information
		if (k % 10 == 0 && k > 0) {
			end = timer.get_time_spent();
			printf("%05d, %6.3f, %6.3f, %6.3f | ETA %9.3fs, %6.3f\n",
				k, q, r, loglik, float(end) / (k) * (numMCMC - k), accept_counter * 100.0f / k);
		}
	}
	time = timer.get_time_spent();
}


// observation
af::array smc::h(const af::array &xt) {
	return af::pow(xt, 2.0) / 20.0;
}

// observation
float smc::h(float xt) {
	return std::pow(xt, 2.0f) / 20.0f;
}

// latent 
float smc::f(float xt, float t) {
	return 0.5f * xt + 25.0f * xt / (1 + std::pow(xt, 2.0f)) + 8 * std::cos(1.2f * t);
}

// latent
af::array smc::f(const af::array &xt, float t) {
	return 0.5f * xt + (25.0f * xt) / (1.0 + af::pow(xt, 2.0f)) + 8 * std::cos(1.2 * t);
}

// resample step
af::array smc::resample_metropolis(const af::array &weights, int iterations) {
	int n = (int)weights.dims(0);
	af::array k = af::seq(0, n - 1);

	for (int i = 0; i < iterations; i++) {
		af::array u = af::randu(n, 1);
		af::array j = af::round(0.5f + (n - 1.5f) * af::randu(n, 1));
		af::array ratio = weights(j) / weights(k);
		k(u <= ratio) = j(u <= ratio);
	}
	return k;
}




// test the smc class
int main(int argc, char *argv[]){
	smc test(2019);
	test.generate_data(0.1f, 10.0f, 3000);
	test.set_mcmc(3000, 1000);
	test.set_param(1.0f, 0.1f);
	test.set_prior(0.01f, 0.01f);
	test.pmmh(0.1f, 0.1f);

	return 0;
}

