cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name G4OK)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(CFG4   REQUIRED CONFIG)   # brings G4 with it 
find_package(OKConf REQUIRED CONFIG)   
find_package(OKOP   REQUIRED CONFIG)
find_package(G4DAE  REQUIRED CONFIG)


set(SOURCES
    G4OpticksManager.cc
    G4OpticksGeometry.cc

    G4OK_LOG.cc
)
set(HEADERS
    G4OpticksManager.hh
    G4OpticksGeometry.hh
    G4OpticksScintillationStep.hh
    G4OpticksCerenkovStep.hh

    G4OK_LOG.hh
    G4OK_API_EXPORT.hh
    G4OK_HEAD.hh
    G4OK_TAIL.hh
)

add_library(${name} SHARED  ${SOURCES})

target_link_libraries( ${name} 
  PUBLIC 
     Opticks::CFG4
     Opticks::G4DAE
     Opticks::OKOP
)
## hmm these need to be public for the tests to see the includes of 
## the dependencies and their deps and so on 

target_compile_definitions( ${name} PUBLIC OPTICKS_G4OK )

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)

