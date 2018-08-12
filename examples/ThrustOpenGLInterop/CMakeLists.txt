cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name ThrustOpenGLInterop)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(FLAGS_VERBOSE ON) 
find_package(OKConf REQUIRED CONFIG)   

set(OpticksGLFW_VERBOSE ON)
find_package(OpticksGLFW REQUIRED MODULE)   
message(STATUS "OpticksGLFW_MODULE  : ${OpticksGLFW_MODULE} ")
echo_target_std(Opticks::OpticksGLFW)

set(OpticksCUDA_VERBOSE ON)
find_package(OpticksCUDA REQUIRED MODULE) 
message(STATUS "OpticksCUDA_MODULE  : ${OpticksCUDA_MODULE} ")


cuda_add_executable( ${name} thrust_opengl_interop.cu ) 

target_link_libraries( ${name} Opticks::OpticksGLFW Opticks::CUDA)

install(TARGETS ${name} DESTINATION lib)

