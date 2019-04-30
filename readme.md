## GPU Implementation for Particle Filter with Metropolis Hasting

This is a GPU implementation of the [Particle Markov chain Monte Carlo methods][1].

### prepare environment

The following packages are needed in this code:

- [x] eigen
- [x] arrayfire
- [x] cmake
- [x] ubuntu 18.04

**Install cmake** 

```shell
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:george-edison55/cmake-3.x
sudo apt-get update
sudo apt-get install cmake
```

**Install Eigen**

```shell
sudo apt-get install libeigen3-dev
```

**Install arrayfire**

1. Install cuda 10.1 from nvidia's [webpage](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu).
2. Install arrayfire 3.6.3 from this [page](<http://arrayfire.org/docs/using_on_linux.htm>).

The step to install *arrayfire* is summarized as follows:

```shell
# download arrayfire
wget https://arrayfire.s3.amazonaws.com/3.6.3/ArrayFire-v3.6.3_Linux_x86_64.sh
# change to excutable
chmod +x ArrayFire-v3.6.3_Linux_x86_64.sh
# install
bash ArrayFire-v3.6.3_Linux_x86_64.sh
# copy extracted file to /opt/
sudo cp -r /arrayfire/ /opt/arrayfire
# add link to cmake
cat /opt/arrayfire/lib64 >> /etc/ld.so.conf.d/arrayfire.conf
ldconfig
```



## Reference

```
[1]: Particle Markov chain Monte Carlo methods, Andrieu, Christophe and Doucet, Arnaud and Holenstein, Roman, Journal of the Royal Statistical Society: Series B (Statistical Methodology)
```

 

