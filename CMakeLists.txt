cmake_minimum_required(VERSION 3.14)
project(test_cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

cuda_add_executable(test_cuda 02_gemm.cu common.h)


