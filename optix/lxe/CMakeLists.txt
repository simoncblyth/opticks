cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name lxe)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

include(EnvBuildOptions)

find_package(Geant4 REQUIRED)  


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Geant4_INCLUDE_DIRS}
)

set(LIBRARIES
   ${Geant4_LIBRARIES}
)

set(DEFINITIONS
   ${Geant4_DEFINITIONS}
)

include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})

add_library(${name} SHARED 

LXeDetectorMessenger.cc
LXeDetectorConstruction.cc
LXeEMPhysics.cc
LXeGeneralPhysics.cc
LXeMuonPhysics.cc
LXePMTSD.cc
LXePhysicsList.cc
LXeScintSD.cc
LXeScintHit.cc
LXePMTHit.cc
LXeMainVolume.cc
LXeWLSSlab.cc
LXeWLSFiber.cc
LXeUserTrackInformation.cc

LXeEventAction.cc
LXeEventMessenger.cc
LXeUserEventInformation.cc
LXePrimaryGeneratorAction.cc
LXeRunAction.cc
LXeStackingAction.cc
LXeSteppingAction.cc
LXeSteppingMessenger.cc
LXeSteppingVerbose.cc
LXeTrackingAction.cc
LXeTrajectory.cc

LXeActionInitialization.cc


)

SET(hdrs 

LXeDetectorMessenger.hh
LXeDetectorConstruction.hh
LXeEMPhysics.hh
LXeGeneralPhysics.hh
LXeMuonPhysics.hh
LXePMTSD.hh
LXePhysicsList.hh
LXeScintSD.hh
LXeScintHit.hh
LXePMTHit.hh
LXeMainVolume.hh
LXeWLSSlab.hh
LXeWLSFiber.hh
LXeUserTrackInformation.hh


LXeEventAction.hh
LXeEventMessenger.hh
LXeUserEventInformation.hh
LXePrimaryGeneratorAction.hh
LXeRunAction.hh
LXeStackingAction.hh
LXeSteppingAction.hh
LXeSteppingMessenger.hh
LXeSteppingVerbose.hh
LXeTrackingAction.hh
LXeTrajectory.hh

LXeActionInitialization.hh
LXeRecorderBase.hh

)
   
target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

add_executable(${name}test ${name}.cc)                  
target_link_libraries(${name}test ${name} ${LIBRARIES})

install(TARGETS ${name}test    DESTINATION bin)


