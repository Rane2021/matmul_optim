cmake_minimum_required(VERSION 3.14)
project(test_cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

cuda_add_executable(test_cuda 03_gemm_register_v1.cu common.h)
target_link_libraries(test_cuda ${CUDA_cublas_LIBRARY})

#  nvcc ../03_gemm_register.cu -o test -lcublas