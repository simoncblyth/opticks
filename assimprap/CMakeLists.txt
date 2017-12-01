cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name AssimpRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(PLog REQUIRED)
find_package(OpticksBoost REQUIRED)

find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(OpticksAssimp REQUIRED)
find_package(GGeo REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)
find_package(OpticksCore REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PLog_INCLUDE_DIRS}  
    ${OpticksBoost_INCLUDE_DIRS}  
    ${SysRap_INCLUDE_DIRS}  
    ${BoostRap_INCLUDE_DIRS}  
    ${OpticksAssimp_INCLUDE_DIRS}  
    ${GGeo_INCLUDE_DIRS}  
    ${GLM_INCLUDE_DIRS}  
    ${NPY_INCLUDE_DIRS}  
    ${OpticksCore_INCLUDE_DIRS}  
)

add_definitions(
    ${PLog_DEFINITIONS}  
    ${OpticksBoost_DEFINITIONS}  
    ${SysRap_DEFINITIONS}  
    ${BoostRap_DEFINITIONS}  
    ${OpticksAssimp_DEFINITIONS} 
    ${GGeo_DEFINITIONS} 
    ${GLM_DEFINITIONS} 
    ${NPY_DEFINITIONS} 
    ${OpticksCore_DEFINITIONS} 
)

set(LIBRARIES
    ${PLog_LIBRARIES}  
    ${OpticksBoost_LIBRARIES} 
    ${SysRap_LIBRARIES} 
    ${BoostRap_LIBRARIES} 
    ${OpticksAssimp_LIBRARIES} 
    ${GGeo_LIBRARIES} 
    ${NPY_LIBRARIES} 
    ${OpticksCore_LIBRARIES} 
)

if(UNIX)
  set(LIBRARIES ${LIBRARIES} ssl crypto)
endif(UNIX)


set(_SOURCES
)


set(SOURCES

    ASIRAP_LOG.cc

    AssimpCommon.cc
    AssimpRegistry.cc
    AssimpSelection.cc
    AssimpTree.cc
    AssimpNode.cc

    AssimpGGeo.cc
    AssimpImporter.cc
)

set(HEADERS

    ASIRAP_LOG.hh
    ASIRAP_API_EXPORT.hh
    ASIRAP_HEAD.hh
    ASIRAP_TAIL.hh


    AssimpCommon.hh
    AssimpGGeo.hh
    AssimpImporter.hh
    AssimpNode.hh
    AssimpRegistry.hh
    AssimpSelection.hh
    AssimpTree.hh
)

add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} ${LIBRARIES})
install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)

