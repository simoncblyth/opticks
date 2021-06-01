cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CFG4)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)  

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()

 

find_package(G4              REQUIRED MODULE)
find_package(ExtG4           REQUIRED CONFIG) 


find_package(OpticksXercesC  REQUIRED MODULE)
find_package(OpticksGeo      REQUIRED CONFIG)
find_package(ThrustRap       REQUIRED CONFIG)   # for TCURAND

#message(STATUS "Geant4_DEFINITIONS:${Geant4_DEFINITIONS} ")
#add_definitions(${Geant4_DEFINITIONS} -DG4USE_STD11)
add_definitions(-DG4USE_STD11)


set(SOURCES
    CFG4_LOG.cc

    PhysicsList.cc
    Scintillation.cc

    DsG4Scintillation.cc
    DsG4OpRayleigh.cc
    DsG4OpBoundaryProcess.cc
    DsG4Cerenkov.cc

    DsPhotonTrackInfo.cc
    DsG4CompositeTrackInfo.cc
    CTrk.cc
    CTrackInfo.cc
    CEventInfo.cc

    CCerenkov.cc
    G4Cerenkov1042.cc
    C4Cerenkov1042.cc
    CPhysicsList.cc 
    CCerenkovGenerator.cc

    CHit.cc
    CSensitiveDetector.cc 


    Cerenkov.cc
    OpRayleigh.cc

    OpNovicePhysicsListMessenger.cc
    OpNovicePhysicsList.cc
    DebugG4Transportation.cc

    SteppingVerbose.cc
    ActionInitialization.cc

    CPhysics.cc
    CGeometry.cc
    CGenerator.cc

    CStep.cc
    CStp.cc
    CPoi.cc
    CRec.cc

    CPrimaryVertex.cc
    CThreeVector.cc
    CEvent.cc
    CTrack.cc
    CPropLib.cc
    CMaterialSort.cc
    CMaterialLib.cc
    CMath.cc
    CSolid.cc
    CMPT.cc
    CMaterial.cc
    CVec.cc
    CMaterialTable.cc
    CMaterialBridge.cc
    CParticleDefinition.cc

    CBorderSurfaceTable.cc
    CSkinSurfaceTable.cc
    CSurfaceTable.cc
    CSurfaceBridge.cc
    COpticalSurface.cc
    CBndLib.cc


    OpStatus.cc
    Format.cc
    CTraverser.cc
    CCheck.cc
    State.cc

    CSource.cc
    CPrimaryGeneratorAction.cc

    CRecorder.cc
    CRecState.cc
    CDebug.cc
    CPhoton.cc
    CAction.cc
    CWriter.cc

    CDetector.cc
    CGDMLDetector.cc
    CGDML.cc
    COptical.cc
    CSurfaceLib.cc
    CMaker.cc
    CTestDetector.cc
    CStepRec.cc
    CGunSource.cc
    CTorchSource.cc
    CInputPhotonSource.cc

    CEventAction.cc
    CSteppingAction.cc
    CBoundaryProcess.cc
    CTrackingAction.cc
    CStage.cc
    CProcessSwitches.cc
    CG4Ctx.cc
    CG4.cc
    CManager.cc

    C4FPEDetection.cc
    CRayTracer.cc
    CVis.cc
    CGenstepCollector.cc

    CGenstepSource.cc
    CPrimarySource.cc
    CPrimaryCollector.cc
    CPhotonCollector.cc
    C4PhotonCollector.cc
    CRunAction.cc
    CEventAction.cc

    CRandomEngine.cc
    CRandomListener.cc
    CAlignEngine.cc
    CMixMaxRng.cc

    CStepping.cc
    CSteppingState.cc
    CStepStatus.cc

    CDump.cc

    CProcess.cc
    CProcessManager.cc
    CProcessSubType.cc
)






set(HEADERS
    CFG4_LOG.hh
    CFG4_API_EXPORT.hh
    CFG4_HEAD.hh
    CFG4_BODY.hh
    CFG4_TAIL.hh
    CFG4_PUSH.hh
    CFG4_POP.hh

    PhysicsList.hh
    Scintillation.hh

    DsG4Scintillation.h
    DsG4OpRayleigh.h
    DsG4OpBoundaryProcess.h
    DsG4OpBoundaryProcessStatus.h
    DsG4Cerenkov.h

    DsPhotonTrackInfo.h
    DsG4CompositeTrackInfo.h

    CTrk.hh
    CTrackInfo.hh
    CEventInfo.hh

    CCerenkov.hh
    G4Cerenkov1042.hh
    C4Cerenkov1042.hh
    CPhysicsList.hh
    CCerenkovGenerator.hh

    CHit.hh
    CSensitiveDetector.hh


    Cerenkov.hh
    OpRayleigh.hh

    OpNovicePhysicsListMessenger.hh
    OpNovicePhysicsList.hh
    DebugG4Transportation.hh

    SteppingVerbose.hh
    ActionInitialization.hh

    CPhysics.hh
    CGeometry.hh
    CGenerator.hh

    CStep.hh
    CStp.hh
    CRec.hh


    CPrimaryVertex.hh
    CThreeVector.hh
    CEvent.hh
    CTrack.hh

    CPropLib.hh
    CMaterialSort.hh
    CMaterialLib.hh

    CMath.hh
    CSolid.hh
    CMPT.hh
    CMaterial.hh
    CVec.hh
    CMaterialTable.hh
    CMaterialBridge.hh
    CParticleDefinition.hh

    CBorderSurfaceTable.hh
    CSkinSurfaceTable.hh
    CSurfaceTable.hh
    CSurfaceBridge.hh
    COpticalSurface.hh
    CBndLib.hh

    OpStatus.hh
    Format.hh
    CTraverser.hh
    State.hh

    CSource.hh
    CPrimaryGeneratorAction.hh

    CRecorder.hh
    CDetector.hh
    CGDMLDetector.hh
    CGDML.hh
    CMaker.hh
    CTestDetector.hh
    CStepRec.hh
    CGunSource.hh
    CTorchSource.hh

    CEventAction.hh
    CSteppingAction.hh
    CTrackingAction.hh
    CStage.hh
    CProcessSwitches.hh
    CG4Ctx.hh
    CG4.hh
    CManager.hh


    C4FPEDetection.hh
    CGenstepCollector.hh

    CGenstepSource.hh
    CPrimarySource.hh
    CPrimaryCollector.hh
    CPhotonCollector.hh
    C4PhotonCollector.hh
    CRunAction.hh
    CEventAction.hh

    CRandomEngine.hh
    CAlignEngine.hh
    CRandomListener.hh
    CMixMaxRng.hh

    CDump.hh

    CProcess.hh
    CProcessManager.hh
    CProcessSubType.hh


)

list(APPEND SOURCES
	CGDMLKludge.cc
	CGDMLKludgeRead.cc
	CGDMLKludgeWrite.cc
)
list(APPEND HEADERS
   	CGDMLKludgeErrorHandler.hh
	CGDMLKludge.hh
	CGDMLKludgeRead.hh
	CGDMLKludgeWrite.hh
)

add_library(${name} SHARED  ${SOURCES})

target_link_libraries( ${name} PUBLIC 
   Opticks::G4
   Opticks::ExtG4
   Opticks::OpticksXercesC
   Opticks::OpticksGeo
   Opticks::ThrustRap
)
target_compile_definitions( ${name} 
   PUBLIC OPTICKS_CFG4 
   PRIVATE WITH_OPTICKS_GENSTEP_COLLECTION
)

target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)

