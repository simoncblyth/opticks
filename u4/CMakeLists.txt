cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name U4)
set(desc "Geant4 Low Level Utilities")
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)

#[=[
U4
===


#]=]

find_package(SysRap REQUIRED CONFIG)
find_package(G4 REQUIRED MODULE)

find_package(CLHEP  REQUIRED CONFIG) 
# Geant4 should have found CLHEP already, but when using Randomize get 
# undefined symbols CLHEP::HepRandom::getTheEngine()


set(SOURCES
    U4_LOG.cc 
    U4.cc
) 

set(HEADERS
    U4_LOG.hh
    U4_API_EXPORT.hh
    U4.hh
    U4Track.h
    U4PhotonInfo.h
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

target_link_libraries( ${name} PUBLIC
    Opticks::G4
    Opticks::SysRap 
    CLHEP::CLHEP
)

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

target_compile_definitions( ${name} PUBLIC OPTICKS_U4 )
target_compile_definitions( ${name} PUBLIC G4USE_STD11 ) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)



