#[=[
UseG4DAE
==========

* access to the INTERFACE of the direct dependancy "G4DAE" is configured here, 
  via the find_package(G4DAE...)

* access to indirect dependancies such as G4,OpticksXercesC 
  are not configured here. Those are automatically hooked up 
  via CONFIG targets imported from::

       $(opticks-prefix)/lib/cmake/*/*-config.cmake   
       $(opticks-prefix)/externals/lib/cmake/*/*-config.cmake

  In this case::

       $(opticks-prefix)/externals/lib/cmake/g4dae/g4dae-config.cmake

  In turn consumers of UseG4DAE (eg UseUseG4DAE) import the targets 
  persisted by the bcm_deploy below to::

       $(opticks-prefix)/lib/cmake/useg4dae/useg4dae-config.cmake 

#]=]


cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseG4DAE)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(G4DAE_VERBOSE ON)
find_package(G4DAE REQUIRED CONFIG)   

message(STATUS "G4DAE_CONFIG  : ${G4DAE_CONFIG} ")
#include(EchoTarget)
#echo_target_std(Opticks::G4DAE)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(SOURCES ${name}.cc)
set(HEADERS ${name}.hh)

add_library(${name} SHARED ${SOURCES})

target_link_libraries(${name} 
  PUBLIC 
     Opticks::G4DAE
)

target_include_directories(${name} 
  PUBLIC
     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

