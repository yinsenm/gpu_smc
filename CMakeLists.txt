# Copyright (c) 2018, ArrayFire
# All rights reserved.
#
# This file is distributed under 3-clause BSD license.
# The complete license agreement can be obtained at:
# http://arrayfire.com/licenses/BSD-3-Clause

cmake_minimum_required(VERSION 3.0)
project(ArrayFire-Example-PF
  VERSION 3.5.0
  LANGUAGES CXX)

find_package(ArrayFire)
find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

if(ArrayFire_CPU_FOUND)
  # Hello World example
  add_executable(pmmh_cpu pmmh.cpp)
  target_link_libraries(pmmh_cpu ArrayFire::afcpu)
  target_link_libraries(pmmh_cpu PUBLIC ${EIGEN3_LIBRARIES})
endif()

if(ArrayFire_CUDA_FOUND)
  add_executable(pmmh_cuda pmmh.cpp)
  target_link_libraries(pmmh_cuda ArrayFire::afcuda)
  target_link_libraries(pmmh_cuda PUBLIC ${EIGEN3_LIBRARIES})
endif()

if(ArrayFire_OpenCL_FOUND)
  add_executable(pmmh_opencl pmmh.cpp)
  target_link_libraries(pmmh_opencl ArrayFire::afopencl)
  target_link_libraries(pmmh_opencl PUBLIC ${EIGEN3_LIBRARIES})
endif()
