cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOKConf)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OKConf_VERBOSE ON)
find_package(OKConf REQUIRED CONFIG)   
message(STATUS "OKConf_DIR : ${OKConf_DIR} ")
message(STATUS "COMPUTE_CAPABILITY : ${COMPUTE_CAPABILITY} ")

add_executable(${name} ${name}.cc )

target_link_libraries(${name} PUBLIC Opticks::OKConf)

install(TARGETS ${name} DESTINATION lib)

