cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseExtG4)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(ExtG4_VERBOSE ON)
find_package(ExtG4 REQUIRED CONFIG)   

message(STATUS "ExtG4_CONFIG  : ${ExtG4_CONFIG} ")
message(STATUS "ExtG4_DIR     : ${ExtG4_DIR} ")

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::ExtG4 )
install(TARGETS ${name}  DESTINATION lib)

