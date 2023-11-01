cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
#[=[
U4
===

Geant4 should have found CLHEP already, but when using Randomize get 
undefined symbols CLHEP::HepRandom::getTheEngine() so also find CLHEP.

Note there there is no find_package Custom4 as that is found when 
finding PMTSim

#]=]

set(name U4)
set(desc "Geant4 Low Level Utilities")
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)

message(STATUS "${CMAKE_CURRENT_LIST_FILE} : ====== 0 ======= find SysRap, G4  ")

find_package(SysRap REQUIRED CONFIG)
find_package(G4 REQUIRED MODULE)

message(STATUS "${CMAKE_CURRENT_LIST_FILE} : ====== 1 ======= find GDXML CLHEP OpticksXercesC ")

find_package(GDXML REQUIRED CONFIG)
find_package(CLHEP  REQUIRED CONFIG) 
find_package(OpticksXercesC REQUIRED MODULE)

message(STATUS "${CMAKE_CURRENT_LIST_FILE} : ====== 2 =======  find Custom4 PMTSim  ")

find_package(Custom4 CONFIG)  # have to be explicit as Custom4 needed without j/PMTSim
find_package(PMTSim CONFIG)  



message(STATUS "${CMAKE_CURRENT_LIST_FILE} : PMTSim_FOUND       : ${PMTSim_FOUND}")
message(STATUS "${CMAKE_CURRENT_LIST_FILE} : PMTSim_VERSION     : ${PMTSim_VERSION}")
message(STATUS "${CMAKE_CURRENT_LIST_FILE} : Custom4_FOUND      : ${Custom4_FOUND}")
message(STATUS "${CMAKE_CURRENT_LIST_FILE} : Custom4_VERSION    : ${Custom4_VERSION}")
message(STATUS "${CMAKE_CURRENT_LIST_FILE} : GDXML_FOUND      : ${GDXML_FOUND}")
message(STATUS "${CMAKE_CURRENT_LIST_FILE} : GDXML_VERSION    : ${GDXML_VERSION}")




message(STATUS "${CMAKE_CURRENT_LIST_FILE} : ====== 3 ======= completed finds " )




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

    InstrumentedG4OpBoundaryProcess.cc
    ShimG4OpAbsorption.cc
    ShimG4OpRayleigh.cc
    Local_G4Cerenkov_modified.cc
    Local_DsG4Scintillation.cc

    U4Physics.cc
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
    Deprecated_U4PhotonInfo.h
    U4GDML.h
    U4Transform.h
    U4Tree.h
    U4TreeBorder.h
    U4Boundary.h
    U4NistManager.h

    U4App.h 
    U4VPrimaryGenerator.h 

    U4SensorIdentifier.h
    U4SensorIdentifierDefault.h

    U4SolidMaker.hh
    U4SolidTree.hh

    U4Material.hh
    U4Mat.h 
    U4Scint.h

    U4Volume.h
    U4Surface.h
    U4Solid.h
    U4Polycone.h
    U4MaterialPropertiesTable.h
    U4MaterialPropertyVector.h

    U4VolumeMaker.hh
    U4Recorder.hh
    U4OpBoundaryProcess.h
    U4StepPoint.hh
    U4Random.hh

    U4Engine.h
    U4Touchable.h

    U4Navigator.h
    U4Simtrace.h 

    U4Debug.hh
    U4Scintillation_Debug.hh
    U4Cerenkov_Debug.hh
    U4Hit_Debug.hh

    InstrumentedG4OpBoundaryProcess.hh
    ShimG4OpAbsorption.hh
    ShimG4OpRayleigh.hh

    Local_G4Cerenkov_modified.hh
    Local_DsG4Scintillation.hh
    U4Physics.hh

    U4PMTAccessor.h 

    U4Process.h
    U4PhysicsTable.h
    U4PhysicsVector.h
    U4MaterialTable.h
    U4OpticalSurfaceFinish.h
    U4OpticalSurfaceModel.h 
    U4SurfaceType.h 
    U4SurfacePerfect.h
    U4SurfaceArray.h 
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

target_compile_definitions( ${name} PUBLIC OPTICKS_U4 ) # THIS MUST BE PUBLIC FOR SLOG-ING TO WORK FULLY 

target_compile_definitions( ${name} PRIVATE G4USE_STD11 ) 
target_compile_definitions( ${name} PRIVATE STANDALONE ) 
target_compile_definitions( ${name} PRIVATE DEBUG_TAG ) 
#target_compile_definitions( ${name} PRIVATE U4Recorder_DEBUG )


if(Custom4_FOUND)
    target_link_libraries( ${name} PUBLIC Custom4::Custom4 )
    target_compile_definitions( ${name} PRIVATE WITH_CUSTOM4 )  # MINTING A DEF ? MAYBE COMES WITH THE TARGET ANYHOW
endif()

if(PMTSim_FOUND)
    target_link_libraries( ${name} PUBLIC Opticks::PMTSim )
    target_compile_definitions( ${name} PRIVATE WITH_PMTSIM PMTSIM_STANDALONE WITH_NP )
    ## THE DEFS CANNOT BE PUBLIC : AS CAUSE ISSUES FOR JUNOSW BUILD WITH OPTICKS
endif()


install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)



