cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name okg4)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

include(EnvBuildOptions)
include(EnvCompilationFlags)

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


find_package(EnvXercesC  REQUIRED)  

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
)

include_directories(${INCLUDE_DIRS})   
add_definitions(${DEFINITIONS})



set(SOURCES
     
    OKG4_LOG.cc

)
set(HEADERS

    OKG4_LOG.hh
    OKG4_API_EXPORT.hh
    OKG4_HEAD.hh
    OKG4_TAIL.hh

)



add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} ${LIBRARIES})
install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

add_subdirectory(tests)


