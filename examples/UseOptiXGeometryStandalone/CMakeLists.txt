cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiXGeometryStandalone)
project(${name})

#[=[
Attempt to do the same as UseOptiXGeometry but without 
depending on an Opticks installation 
#]=]


if(POLICY CMP0077)  # see notes/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


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

add_library(ImpGLM INTERFACE IMPORTED) 
set_target_properties(ImpGLM  PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/externals/glm/glm"
)

set(COMPUTE_CAPABILITY "70" CACHE STRING "Compute capability")
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
    ${name}.cu
    box.cu
)

CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${OPTIX_SOURCES} )
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )

add_executable( ${name} ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )

target_link_libraries(${name} PUBLIC 
      ImpGLM 
      ImpOptiX 
      ImpCUDA 
)

install(FILES ${_generated_PTX_files} DESTINATION ptx)
install(TARGETS ${name}               DESTINATION bin)


