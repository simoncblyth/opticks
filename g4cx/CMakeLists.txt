cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name G4CX)
set(desc "Top Level")
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)

#[=[
G4CX
=====

Note : this must remain a franken-package until rework the 
geometry chain into using a more direct (and shallower) 
new world approach. 

#]=]

#find_package(SysRap REQUIRED CONFIG)
#find_package(G4 REQUIRED MODULE)
#find_package(CLHEP  REQUIRED CONFIG) 
# Geant4 should have found CLHEP already, but when using Randomize get 
# undefined symbols CLHEP::HepRandom::getTheEngine()


find_package(U4       REQUIRED CONFIG)
find_package(ExtG4    REQUIRED CONFIG)
find_package(CSG_GGeo REQUIRED CONFIG)
find_package(CSGOptiX REQUIRED CONFIG)


set(SOURCES
    G4CX_LOG.cc 

    G4CXOpticks.cc
) 

set(HEADERS
    G4CX_LOG.hh
    G4CX_API_EXPORT.hh

    G4CXOpticks.hh
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

target_link_libraries( ${name} PUBLIC
    Opticks::U4
    Opticks::ExtG4
    Opticks::CSG_GGeo
    Opticks::CSGOptiX
)

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

target_compile_definitions( ${name} PUBLIC OPTICKS_G4CX )
target_compile_definitions( ${name} PUBLIC G4USE_STD11 ) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)



