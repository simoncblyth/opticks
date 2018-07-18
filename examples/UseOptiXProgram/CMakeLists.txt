cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiXProgram)
project(${name})
include(OpticksBuildOptions)

find_package(OKConf REQUIRED CONFIG)   
include(OpticksCompilationFlags)     ## repeating this here as nvcc flags require COMPUTE_CAPABILIY from OKConf     
find_package(OptiX MODULE QUIET)   
find_package(OpticksCUDA REQUIRED MODULE) 

set(SOURCES
    UseOptiXProgram.cc
)

set(OPTIX_SOURCES
    UseOptiXProgram_minimal.cu
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

