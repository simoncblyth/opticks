cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
#[=[
Y4CSG
=======

Tightly focussed package : aiming to convert a 
pure Geant4 CSG shape defined by a combination of 
G4VSolids into an Opticks NCSG instance for serialization. 

Package needs to be usuable with minimal code from 
code generated 


#]=]

set(name Y4CSG)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(G4  REQUIRED MODULE)
find_package(NPY REQUIRED CONFIG)

set(SOURCES
    Y4CSG.cc
    Y4CSG_LOG.cc
)
set(HEADERS
    Y4CSG.hh
    Y4CSG_LOG.hh
    Y4CSG_API_EXPORT.hh
    Y4CSG_HEAD.hh
    Y4CSG_TAIL.hh
)


add_library(${name} SHARED  ${SOURCES})

target_link_libraries( ${name} 
  PUBLIC 
     Opticks::G4
     Opticks::NPY
)


target_compile_definitions( ${name} PUBLIC OPTICKS_Y4CSG )

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)

