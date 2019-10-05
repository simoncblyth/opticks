cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksXercesC)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OpticksXercesC_VERBOSE ON)
find_package(OpticksXercesC REQUIRED MODULE)   

message(STATUS "OpticksXercesC_MODULE  : ${OpticksXercesC_MODULE} ")

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OpticksXercesC )

install(TARGETS ${name}  DESTINATION lib)

