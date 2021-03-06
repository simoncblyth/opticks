cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(name ThrustOpenGLInterop)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


set(FLAGS_VERBOSE ON) 
find_package(OKConf REQUIRED CONFIG)   


find_package(OpticksGLEW REQUIRED MODULE)


set(OpticksGLFW_VERBOSE ON)
find_package(OpticksGLFW REQUIRED MODULE)   
message(STATUS "OpticksGLFW_MODULE  : ${OpticksGLFW_MODULE} ")
echo_target_std(Opticks::OpticksGLFW)

set(OpticksCUDA_VERBOSE ON)
find_package(OpticksCUDA REQUIRED MODULE) 
message(STATUS "OpticksCUDA_MODULE  : ${OpticksCUDA_MODULE} ")


cuda_add_executable( ${name} thrust_opengl_interop.cu ) 

#set(SRC thrust_opengl_interop.cc)
#CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SRC} )
#add_executable( ${name} ${_generated_OBJ_files} ${SRC} )


target_link_libraries( ${name} 
     Opticks::OpticksGLEW
     Opticks::OpticksGLFW 
     Opticks::CUDA
)


install(TARGETS ${name} DESTINATION lib)

