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

#message("${name}.CMAKE_CXX_FLAGS (fg4)  : ${CMAKE_CXX_FLAGS} ")


find_package(EnvXercesC  REQUIRED)  
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
    Cerenkov.cc
    OpRayleigh.cc

    OpNovicePhysicsListMessenger.cc
    OpNovicePhysicsList.cc

    SteppingVerbose.cc
    ActionInitialization.cc

    CStep.cc
    CTrack.cc
    CPropLib.cc
    CMath.cc
    CSolid.cc
    CMPT.cc
    OpStatus.cc
    Format.cc
    CTraverser.cc
    State.cc

    CSource.cc
    CPrimaryGeneratorAction.cc

    CRecorder.cc
    CDetector.cc
    CGDMLDetector.cc
    CMaker.cc
    CTestDetector.cc
    Rec.cc
    CStepRec.cc
    CGunSource.cc
    CTorchSource.cc

    CSteppingAction.cc
    CG4.cc
    OpticksG4Collector.cc
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
    Cerenkov.hh
    OpRayleigh.hh

    OpNovicePhysicsListMessenger.hh
    OpNovicePhysicsList.hh

    SteppingVerbose.hh
    ActionInitialization.hh

    CStep.hh
    CTrack.hh
    CPropLib.hh
    CMath.hh
    CSolid.hh
    CMPT.hh
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
    Rec.hh
    CStepRec.hh
    CGunSource.hh
    CTorchSource.hh

    CSteppingAction.hh
    CG4.hh
    OpticksG4Collector.hh
)

add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} ${LIBRARIES})
install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

add_subdirectory(tests)


