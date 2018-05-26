cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseSysRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(SysRap REQUIRED CONFIG)   
message(STATUS "SysRap_DIR : ${SysRap_DIR} ")

set(SOURCES ${name}.cc )
set(HEADERS ${name}.hh )

add_library(${name} SHARED ${SOURCES})

target_link_libraries(${name} PUBLIC Opticks::SysRap)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

echo_target_std(${name})

