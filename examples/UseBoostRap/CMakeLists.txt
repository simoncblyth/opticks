cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseBoostRap)
project(${name})
include(OpticksBuildOptions)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(BoostRap REQUIRED CONFIG)   
message(STATUS "BoostRap_CONFIG : ${BoostRap_CONFIG} ")

add_executable(${name} ${name}.cc)

target_link_libraries(${name} PUBLIC Opticks::BoostRap)

install(TARGETS ${name}  DESTINATION lib)

