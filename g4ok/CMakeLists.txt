cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name G4OK)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(G4     REQUIRED MODULE)

find_package(OKConf REQUIRED CONFIG)   
find_package(OKOP   REQUIRED CONFIG)
find_package(G4DAE  REQUIRED CONFIG)


set(SOURCES
    G4OKMgr.cc

    G4OK_LOG.cc
)
set(HEADERS
    G4OKMgr.hh

    G4OK_LOG.hh
    G4OK_API_EXPORT.hh
    G4OK_HEAD.hh
    G4OK_TAIL.hh
)

add_library(${name} SHARED  ${SOURCES})

target_link_libraries( ${name} 
  PUBLIC 
     Opticks::G4
  PRIVATE
     Opticks::G4DAE
     Opticks::OKOP
)
target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

#add_subdirectory(tests)

