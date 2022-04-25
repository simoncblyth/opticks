cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseGeometryShader)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OpticksGLFW_ALT ON)
set(OpticksGLFW_VERBOSE ON)

#include_directories($ENV{OPTICKS_PREFIX}/include/SysRap)

string(REPLACE ":" ";" INCLUDE_LIST $ENV{HOME}/np)
include_directories(${INCLUDE_LIST})


find_package(GLM         REQUIRED MODULE)
find_package(OpticksGLEW REQUIRED MODULE)
find_package(OpticksGLFW REQUIRED MODULE)   

message(STATUS "OpticksGLFW_MODULE  : ${OpticksGLFW_MODULE} ")
echo_target_std(Opticks::OpticksGLFW)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} 
  PUBLIC 
    Opticks::GLM 
    Opticks::OpticksGLEW 
    Opticks::OpticksGLFW 
)
install(TARGETS ${name}  DESTINATION lib)

