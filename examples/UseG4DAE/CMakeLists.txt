cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseG4DAE)
project(${name} VERSION 0.1.0)

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")
find_package(BCM)
include(BCMDeploy)

#[=[

Note that the dependancies of G4DAE, 
(G4,  
do not need to be configured here 


#]=]

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

