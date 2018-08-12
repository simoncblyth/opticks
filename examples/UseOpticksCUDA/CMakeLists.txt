cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksCUDA)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(FLAGS_VERBOSE ON) 
set(OpticksCUDA_VERBOSE ON)

find_package(OKConf REQUIRED CONFIG)   
find_package(OpticksCUDA REQUIRED MODULE) 

cuda_add_executable(${name} ${name}.cu)
target_link_libraries(${name} Opticks::CUDA Opticks::CUDASamples)

install(TARGETS ${name} DESTINATION lib)

