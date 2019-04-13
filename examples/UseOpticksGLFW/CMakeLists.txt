cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksGLFW)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OpticksGLFW_ALT ON)
set(OpticksGLFW_VERBOSE ON)
find_package(OpticksGLFW REQUIRED MODULE)   

message(STATUS "OpticksGLFW_MODULE  : ${OpticksGLFW_MODULE} ")
echo_target_std(Opticks::OpticksGLFW)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OpticksGLFW )
install(TARGETS ${name}  DESTINATION lib)

