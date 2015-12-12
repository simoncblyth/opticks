cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name cfg4)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

include(EnvBuildOptions)


find_package(Boost REQUIRED)
find_package(Bregex REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)

find_package(Geant4 REQUIRED)  


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS}
   ${Bregex_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS}
   ${Geant4_INCLUDE_DIRS}
)

set(LIBRARIES
   ${Boost_LIBRARIES}
   ${Bregex_LIBRARIES}
   ${NPY_LIBRARIES}
   ${Geant4_LIBRARIES}
)

set(DEFINITIONS
   ${Boost_DEFINITIONS} 
   ${Bregex_DEFINITIONS} 
   ${Geant4_DEFINITIONS}
)

include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})

add_library(${name} SHARED 
Format.cc
Recorder.cc
PhysicsList.cc
DetectorConstruction.cc
PrimaryGeneratorAction.cc
ActionInitialization.cc
SteppingAction.cc
SteppingVerbose.cc
)

SET(hdrs 
Format.hh
Recorder.hh
PhysicsList.hh
DetectorConstruction.hh
PrimaryGeneratorAction.hh
ActionInitialization.hh
SteppingAction.hh
SteppingVerbose.hh
)
   
target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

add_executable(${name}test ${name}.cc)                  
target_link_libraries(${name}test ${name} ${LIBRARIES})

install(TARGETS ${name}test    DESTINATION bin)


