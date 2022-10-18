cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name U4)
set(desc "Geant4 Low Level Utilities")
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)

#[=[
U4
===

HMM: depending on CSG too would allow loading CSGFoundry ? 
Yes: a possibility, but that is not needed for primary functionality currently
although it would be useful for testing ... 
so depend on that in specific tests not generally 
 

#]=]

find_package(SysRap REQUIRED CONFIG)
find_package(G4 REQUIRED MODULE)
find_package(GDXML REQUIRED CONFIG)

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

    U4Debug.cc
    U4Scintillation_Debug.cc
    U4Cerenkov_Debug.cc
    U4Hit_Debug.cc
    U4FastSim_Debug.cc

    InstrumentedG4OpBoundaryProcess.cc

) 

set(HEADERS
    U4_LOG.hh
    U4_API_EXPORT.hh
    U4.hh
    U4ThreeVector.h
    U4Hit.h 
    U4HitGet.h 
    U4Track.h
    U4Stack.h
    U4PhotonInfo.h
    U4GDML.h
    U4Transform.h
    U4Tree.h

    U4SensorIdentifier.h
    U4SensorIdentifierDefault.h

    U4SolidMaker.hh
    U4SolidTree.hh
    U4Material.hh
    U4VolumeMaker.hh
    U4Recorder.hh
    U4OpBoundaryProcess.h
    U4StepPoint.hh
    U4Random.hh
    U4Touchable.h

    U4Debug.hh
    U4Scintillation_Debug.hh
    U4Cerenkov_Debug.hh
    U4Hit_Debug.hh
    U4FastSim_Debug.hh

    InstrumentedG4OpBoundaryProcess.hh
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

target_link_libraries( ${name} PUBLIC
    Opticks::G4
    Opticks::SysRap 
    CLHEP::CLHEP
    Opticks::OpticksXercesC
    Opticks::GDXML
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



