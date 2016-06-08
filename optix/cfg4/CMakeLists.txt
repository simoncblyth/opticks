cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name cfg4)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

include(EnvBuildOptions)

find_package(OpticksBoost REQUIRED)
find_package(BoostRap REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)
find_package(GGeo REQUIRED)
find_package(OpticksCore REQUIRED)

find_package(Geant4 REQUIRED)  
find_package(EnvXercesC  REQUIRED)  
find_package(G4DAE  REQUIRED)  

set(INCLUDE_DIRS  
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${OpticksBoost_INCLUDE_DIRS}
   ${BoostRap_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}
   ${GGeo_INCLUDE_DIRS}
   ${OpticksCore_INCLUDE_DIRS}
   ${Geant4_INCLUDE_DIRS}
   ${XERCESC_INCLUDE_DIR}
   ${G4DAE_INCLUDE_DIRS}
)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
   ${BoostRap_LIBRARIES}
   ${NPY_LIBRARIES}
   ${GGeo_LIBRARIES}
   ${OpticksCore_LIBRARIES}
   ${Geant4_LIBRARIES}
   ${XERCESC_LIBRARIES}
   ${G4DAE_LIBRARIES}
)

set(DEFINITIONS
   ${OpticksBoost_DEFINITIONS} 
   ${BoostRap_DEFINITIONS} 
   ${NPY_DEFINITIONS} 
   ${GGeo_DEFINITIONS} 
   ${OpticksCore_DEFINITIONS}
   ${Geant4_DEFINITIONS}
   ${XERCESC_DEFINITIONS}
   ${G4DAE_DEFINITIONS}
)

#message(STATUS "XERCESC_INCLUDE_DIR  : ${XERCESC_INCLUDE_DIR}")
#message(STATUS "XERCESC_LIBRARIES    : ${XERCESC_LIBRARIES}")
#message(STATUS "XERCESC_DEFINITIONS  : ${XERCESC_DEFINITIONS}")


include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})

add_library(${name} SHARED 
CG4.cc
CMPT.cc
CTrack.cc
CSolid.cc
CMath.cc
Format.cc
CRecorder.cc
Rec.cc
State.cc
CTorchSource.cc
CGunSource.cc
CSource.cc
CStep.cc
CStepRec.cc
OpStatus.cc
PhysicsList.cc
OpNovicePhysicsList.cc
OpNovicePhysicsListMessenger.cc
OpRayleigh.cc
Scintillation.cc
CDetector.cc
CTestDetector.cc
CGDMLDetector.cc
CMaker.cc
CPropLib.cc
CTraverser.cc
CPrimaryGeneratorAction.cc
ActionInitialization.cc
CSteppingAction.cc
SteppingVerbose.cc
)

SET(hdrs 
CG4.hh
CMPT.hh
CTrack.hh
CSolid.hh
CMath.hh
Format.hh
CRecorder.hh
Rec.hh
State.hh
CTorchSource.hh
CGunSource.hh
CSource.hh
CStep.hh
CStepRec.hh
OpStatus.hh
PhysicsList.hh
OpNovicePhysicsList.hh
OpNovicePhysicsListMessenger.hh
OpRayleigh.hh
Scintillation.hh
CDetector.hh
CTestDetector.hh
CGDMLDetector.hh
CMaker.hh
CPropLib.hh
CTraverser.hh
CPrimaryGeneratorAction.hh
ActionInitialization.hh
CSteppingAction.hh
SteppingVerbose.hh
)
   
target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

add_subdirectory(tests)


