cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(name UseOpticksCUDA)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


set(FLAGS_VERBOSE ON) 
set(OpticksCUDA_VERBOSE ON)

find_package(OKConf REQUIRED CONFIG)   
find_package(OpticksCUDA REQUIRED MODULE) 

cuda_add_executable(${name} ${name}.cu)
target_link_libraries(${name} Opticks::CUDA Opticks::CUDASamples)

install(TARGETS ${name} DESTINATION lib)

