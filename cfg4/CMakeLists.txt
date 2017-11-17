cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name cfg4)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

#message("${name}.CMAKE_CXX_FLAGS (init) : ${CMAKE_CXX_FLAGS} ")
include(EnvBuildOptions)
#message("${name}.CMAKE_CXX_FLAGS (ebo)  : ${CMAKE_CXX_FLAGS} ")
include(EnvCompilationFlags)
#message("${name}.CMAKE_CXX_FLAGS (ecf)  : ${CMAKE_CXX_FLAGS} ")


find_package(PLog REQUIRED)
find_package(OpticksBoost REQUIRED)

find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)
find_package(GGeo REQUIRED)
find_package(OpticksCore REQUIRED)
find_package(OpticksGeometry REQUIRED)

find_package(Geant4 REQUIRED) ## invoke cmake with -DGeant4_DIR=$(g4-cmake-dir) 



# Parsing G4Version.hh header makes the CFG4_G4VERSION_NUMBER
# available at configure time, the value is written to inc/OpticksCMakeConfig.hh 
# by top level CMakeLists.txt

set(CFG4_G4VERSION_NUMBER 0 PARENT_SCOPE)
if(Geant4_FOUND)
   set(_dirs ${Geant4_INCLUDE_DIRS}) 
   list(GET _dirs 0 _firstdir)
   #message(STATUS "${name}.Geant4_INCLUDE_DIRS    : ${Geant4_INCLUDE_DIRS} ")
   #message(STATUS "${name}._firstdir              : ${_firstdir} ")
   file(READ "${_firstdir}/G4Version.hh" _contents)
   #message(STATUS "${name}.contents : ${_contents} ")
   string(REGEX REPLACE "\n" ";" _contents "${_contents}")
   foreach(_line ${_contents})
        if (_line MATCHES "#define G4VERSION_NUMBER[ ]+([0-9]+)$")
            set(CFG4_G4VERSION_NUMBER ${CMAKE_MATCH_1} PARENT_SCOPE)
            message(STATUS "${name}._line ${_line} ===> ${CMAKE_MATCH_1} ") 
        endif()
   endforeach()
else(Geant4_FOUND)
   message(STATUS "${name}.Geant4 NOT-FOUND ")
endif(Geant4_FOUND)








#message("${name}.CMAKE_CXX_FLAGS (fg4)  : ${CMAKE_CXX_FLAGS} ")


find_package(EnvXercesC  REQUIRED)  

message("${name}.XERCESC_INCLUDE_DIR  : ${XERCESC_INCLUDE_DIR} ")
message("${name}.XERCESC_LIBRARIES    : ${XERCESC_LIBRARIES} ")



#find_package(G4DAE  REQUIRED)  

set(INCLUDE_DIRS  
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${PLog_INCLUDE_DIRS}
   ${OpticksBoost_INCLUDE_DIRS}
   ${SysRap_INCLUDE_DIRS} 
   ${BoostRap_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}
   ${GGeo_INCLUDE_DIRS}
   ${OpticksCore_INCLUDE_DIRS}
   ${OpticksGeometry_INCLUDE_DIRS}
   ${Geant4_INCLUDE_DIRS}
   ${XERCESC_INCLUDE_DIR}
 #  ${G4DAE_INCLUDE_DIRS}
)

set(LIBRARIES
   ${PLog_LIBRARIES}
   ${OpticksBoost_LIBRARIES}
   ${SysRap_LIBRARIES}
   ${BoostRap_LIBRARIES}
   ${NPY_LIBRARIES}
   ${GGeo_LIBRARIES}
   ${OpticksCore_LIBRARIES}
   ${OpticksGeometry_LIBRARIES}
   ${Geant4_LIBRARIES}
   ${XERCESC_LIBRARIES}
  # ${G4DAE_LIBRARIES}
)

set(DEFINITIONS
   ${PLog_DEFINITIONS} 
   ${OpticksBoost_DEFINITIONS} 
   ${SysRap_DEFINITIONS} 
   ${BoostRap_DEFINITIONS} 
   ${NPY_DEFINITIONS} 
   ${GGeo_DEFINITIONS} 
   ${OpticksCore_DEFINITIONS}
   ${OpticksGeometry_DEFINITIONS}
   ${Geant4_DEFINITIONS}
   ${XERCESC_DEFINITIONS}
  #${G4DAE_DEFINITIONS}
)

#message(STATUS "XERCESC_INCLUDE_DIR  : ${XERCESC_INCLUDE_DIR}")
#message(STATUS "XERCESC_LIBRARIES    : ${XERCESC_LIBRARIES}")
#message(STATUS "XERCESC_DEFINITIONS  : ${XERCESC_DEFINITIONS}")

include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})


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
    CTrack.cc
    CPropLib.cc
    CMaterialLib.cc
    CMath.cc
    CSolid.cc
    CMPT.cc
    CVec.cc
    CMaterialTable.cc
    CMaterialBridge.cc

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
    CG4Ctx.cc
    CG4.cc
    CCollector.cc
    CRunAction.cc
    CEventAction.cc
)

set(HEADERS
    CFG4_LOG.hh
    CFG4_API_EXPORT.hh
    CFG4_HEAD.hh
    CFG4_TAIL.hh
    CFG4_PUSH.hh
    CFG4_POP.hh

    PhysicsList.hh
    Scintillation.hh

    DsG4Scintillation.h
    DsG4OpRayleigh.h
    DsG4OpBoundaryProcess.h
    DsG4Cerenkov.h
    DsPhotonTrackInfo.h
    DsG4CompositeTrackInfo.h

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
    CTrack.hh

    CPropLib.hh
    CMaterialLib.hh

    CMath.hh
    CSolid.hh
    CMPT.hh
    CVec.hh
    CMaterialTable.hh
    CMaterialBridge.hh

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
    CMaker.hh
    CTestDetector.hh
    CStepRec.hh
    CGunSource.hh
    CTorchSource.hh

    CEventAction.hh
    CSteppingAction.hh
    CTrackingAction.hh
    CStage.hh
    CG4.hh
    CCollector.hh
    CRunAction.hh
    CEventAction.hh
)

add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} ${LIBRARIES})
install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

add_subdirectory(tests)


