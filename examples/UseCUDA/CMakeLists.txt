cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseCUDA)
project(${name} VERSION 0.1.0)

include(EnvBuildOptions)
include(EnvCompilationFlags)

#find_package(CUDA ${OPTICKS_CUDA_VERSION} REQUIRED )
find_package(CUDA REQUIRED )

message(STATUS "${name}.CUDA_VERSION:${CUDA_VERSION}")
message(STATUS "${name}.CUDA_curand_LIBRARY:${CUDA_curand_LIBRARY}")
message(STATUS "${name}.CUDA_LIBRARIES : ${CUDA_LIBRARIES} ")

#[=[
See issues/cuda-thrust-optix-cmake-update.rst
#]=]

