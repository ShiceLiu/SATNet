cmake_minimum_required(VERSION 2.8)  
 
project(datautil)

set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_NVCC_FLAGS -v -std=c++11 -g -Xcompiler -fexceptions -Xcompiler -fPIC)
set(CMAKE_C_FLAGS "-g -fPIC")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++11 -g -fPIC")
set(CUDA_SEPARABLE_COMPILATION ON)
 
find_package(CUDA REQUIRED)
 
set(GENCODE -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=compute_35 -gencode=arch=compute_20,code=sm_20)

include_directories(${CUDA_INCLUDE_DIRS} /usr/include/python2.7)

cuda_add_library(datautil SHARED datautil.cu)

target_link_libraries(datautil ${CUDA_LIBRARIES})