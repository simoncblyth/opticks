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
find_package(OpticksXercesC REQUIRED MODULE)

find_package(PMTSim CONFIG)


set(SOURCES
    U4_LOG.cc 
    U4.cc
    U4SolidMaker.cc
    U4SolidTree.cc
    U4Material.cc
    U4VolumeMaker.cc
    U4Recorder.cc
    U4StepPoint.cc
    U4Random.cc
    InstrumentedG4OpBoundaryProcess.cc

) 

set(HEADERS
    U4_LOG.hh
    U4_API_EXPORT.hh
    U4.hh
    U4Hit.h 
    U4Track.h
    U4Stack.h
    U4PhotonInfo.h
    U4GDML.h
    U4SolidMaker.hh
    U4SolidTree.hh
    U4Material.hh
    U4VolumeMaker.hh
    U4Recorder.hh
    U4OpBoundaryProcess.h
    U4StepPoint.hh
    U4Random.hh
    InstrumentedG4OpBoundaryProcess.hh
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

target_link_libraries( ${name} PUBLIC
    Opticks::G4
    Opticks::SysRap 
    CLHEP::CLHEP
    Opticks::OpticksXercesC
)

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

target_compile_definitions( ${name} PUBLIC OPTICKS_U4 )
target_compile_definitions( ${name} PUBLIC G4USE_STD11 ) 
target_compile_definitions( ${name} PUBLIC STANDALONE ) 
#target_compile_definitions( ${name} PRIVATE DEBUG_PIDX ) 
target_compile_definitions( ${name} PUBLIC DEBUG_TAG ) 


if(PMTSim_FOUND)
    target_link_libraries( ${name} PUBLIC Opticks::PMTSim )
    target_compile_definitions( ${name} PUBLIC WITH_PMTSIM )
endif()



install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)



