cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiXProgram)
project(${name})
include(OpticksBuildOptions)

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()




find_package(OKConf      REQUIRED CONFIG)   
find_package(OptiX       REQUIRED MODULE)   
find_package(OpticksCUDA REQUIRED MODULE) 

set(SOURCES
    UseOptiXProgram.cc
)

set(OPTIX_SOURCES
    minimal.cu
)

CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${OPTIX_SOURCES} )
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )

add_executable( ${name} ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )

target_link_libraries(${name} PUBLIC Opticks::OptiX Opticks::OKConf)

install(FILES ${_generated_PTX_files} DESTINATION installcache/PTX)
install(TARGETS ${name}  DESTINATION lib)

#[=[
This is intended to be a minimal example of OptiX program usage 
#]=]

