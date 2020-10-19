cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiXFan)
project(${name})

#[=[
Turn https://github.com/Adiolph/OptiX-test into an opticks example
using just OptiX 6, with no use of Opticks libs/headers.

Adapted the CMakeLists.txt from ~/opticks/examples/UseOptiXGeometryStandalone/
removing GLM dependency.

#]=]


if(POLICY CMP0077)  # see notes/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


set(CMAKE_CXX_STANDARD 14) 
set(CMAKE_CXX_STANDARD_REQUIRED on) 

set(CMAKE_INSTALL_RPATH ${OptiX_INSTALL_DIR}/lib64)


find_package(OptiX REQUIRED MODULE)   
add_library(ImpOptiX INTERFACE IMPORTED) 
set_target_properties(ImpOptiX PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OptiX_INCLUDE}"
    INTERFACE_LINK_LIBRARIES "optix;optixu;optix_prime"
)

message( STATUS "OptiX_INCLUDE : ${OptiX_INCLUDE} ")


find_package(CUDA REQUIRED MODULE) 
add_library(ImpCUDA INTERFACE IMPORTED)
set_target_properties(ImpCUDA  PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${CUDA_cudart_static_LIBRARY};${CUDA_curand_LIBRARY}"
)


set(COMPUTE_CAPABILITY "30" CACHE STRING "Compute capability")
message(STATUS " COMPUTE_CAPABILITY=${COMPUTE_CAPABILITY} ")

set(CUDA_NVCC_FLAGS)
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
list(APPEND CUDA_NVCC_FLAGS "-O2")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")


include_directories( ${OptiX_INCLUDE} )  
## needed for compiling the PTX, as seems no way to pass in a target


set(SOURCES
    ${name}.cc
)

set(OPTIX_SOURCES
    point_source.cu
    simple_dom.cu
    sphere.cu
)

CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${OPTIX_SOURCES} )
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )

add_executable( ${name} ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )

target_link_libraries(${name} PUBLIC 
      ImpOptiX 
      ImpCUDA 
)

install(FILES ${_generated_PTX_files} DESTINATION ptx)
install(TARGETS ${name}               DESTINATION bin)


