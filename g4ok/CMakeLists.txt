cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name G4OK)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    message(STATUS "with-policy-CMP0077") 
    cmake_policy(SET CMP0077 OLD)
    #cmake_policy(SET CMP0077 NEW)
else()
    message(STATUS "no-policy-CMP0077") 
endif()


find_package(CFG4   REQUIRED CONFIG)   
find_package(ExtG4  REQUIRED CONFIG)   
find_package(OKOP   REQUIRED CONFIG)


set(SOURCES
    G4Opticks.cc
    G4OpticksHit.cc
    G4OpticksRecorder.cc
    G4OK_LOG.cc
)
set(HEADERS
    G4Opticks.hh
    G4OpticksHit.hh
    G4OpticksRecorder.hh

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
     Opticks::ExtG4
     Opticks::OKOP
)

#[=[

Opticks::G4DAE

1. hmm these need to be public for the tests to see the includes of 
   the dependencies and their deps and so on 

#]=]


target_compile_definitions( ${name} PUBLIC OPTICKS_G4OK )

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)

