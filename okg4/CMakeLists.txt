cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OKG4)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(OK REQUIRED CONFIG)
find_package(CFG4 REQUIRED CONFIG)


set(SOURCES
    OKG4Mgr.cc

    OKG4_LOG.cc
)
set(HEADERS
    OKG4Mgr.hh

    OKG4_LOG.hh
    OKG4_API_EXPORT.hh
    OKG4_HEAD.hh
    OKG4_TAIL.hh
)

add_library(${name} SHARED  ${SOURCES})

target_link_libraries( ${name} PUBLIC 
   Opticks::OK
   Opticks::CFG4
)
target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)

