cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseCSGBSP)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(CSGBSP REQUIRED CONFIG)   

message(STATUS "CSGBSP_DIR   : ${CSGBSP_DIR} ")
message(STATUS "CSGBSP_FOUND : ${CSGBSP_FOUND} ")
echo_target_std(Opticks::CSGBSP)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::CSGBSP)

install(TARGETS ${name}  DESTINATION lib)

