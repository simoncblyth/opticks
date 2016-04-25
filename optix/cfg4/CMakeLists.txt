cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name cfg4)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

include(EnvBuildOptions)

find_package(Boost REQUIRED)
find_package(BRegex REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)
find_package(BCfg REQUIRED)
find_package(GGeo REQUIRED)
find_package(OpticksCore REQUIRED)

find_package(Geant4 REQUIRED)  
find_package(EnvXercesC  REQUIRED)  
find_package(G4DAE  REQUIRED)  

set(INCLUDE_DIRS  
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS}
   ${BRegex_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}
   ${BCfg_INCLUDE_DIRS}
   ${GGeo_INCLUDE_DIRS}
   ${OpticksCore_INCLUDE_DIRS}
   ${Geant4_INCLUDE_DIRS}
   ${XERCESC_INCLUDE_DIR}
   ${G4DAE_INCLUDE_DIRS}
)

set(LIBRARIES
   ${Boost_LIBRARIES}
   ${BRegex_LIBRARIES}
   ${NPY_LIBRARIES}
   ${BCfg_LIBRARIES}
   ${GGeo_LIBRARIES}
   ${OpticksCore_LIBRARIES}
   ${Geant4_LIBRARIES}
   ${XERCESC_LIBRARIES}
   ${G4DAE_LIBRARIES}
)

set(DEFINITIONS
   ${Boost_DEFINITIONS} 
   ${BRegex_DEFINITIONS} 
   ${NPY_DEFINITIONS} 
   ${GGeo_DEFINITIONS} 
   ${BCfg_DEFINITIONS} 
   ${OpticksCore_DEFINITIONS}
   ${Geant4_DEFINITIONS}
   ${XERCESC_DEFINITIONS}
   ${G4DAE_DEFINITIONS}
)

message(STATUS "XERCESC_INCLUDE_DIR  : ${XERCESC_INCLUDE_DIR}")
message(STATUS "XERCESC_LIBRARIES    : ${XERCESC_LIBRARIES}")
message(STATUS "XERCESC_DEFINITIONS  : ${XERCESC_DEFINITIONS}")


include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})

add_library(${name} SHARED 
CfG4.cc
Format.cc
Recorder.cc
Rec.cc
State.cc
CSource.cc
OpStatus.cc
PhysicsList.cc
CDetector.cc
CMaker.cc
CPropLib.cc
CTraverser.cc
PrimaryGeneratorAction.cc
ActionInitialization.cc
SteppingAction.cc
SteppingVerbose.cc
)

SET(hdrs 
CfG4.hh
Format.hh
Recorder.hh
Rec.hh
State.hh
CSource.hh
OpStatus.hh
PhysicsList.hh
CDetector.hh
CMaker.hh
CPropLib.hh
CTraverser.hh
PrimaryGeneratorAction.hh
ActionInitialization.hh
SteppingAction.hh
SteppingVerbose.hh
)
   
target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

add_subdirectory(tests)


