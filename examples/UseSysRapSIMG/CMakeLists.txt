cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseSysRapSIMG)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(SysRap REQUIRED CONFIG)   
message(STATUS "SysRap_DIR : ${SysRap_DIR} ")

add_executable(${name} ${name}.cc)

target_link_libraries(${name} PUBLIC Opticks::SysRap)

install(TARGETS ${name} DESTINATION lib)


