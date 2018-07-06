cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseSymbol)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

add_library(${name} SHARED ${name}.cc)

set(HEADERS ${name}.hh)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})



