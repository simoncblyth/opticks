cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name AssimpRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost REQUIRED)
find_package(Assimp REQUIRED)
find_package(GGeo REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)
find_package(OpticksCore REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}

    ${OpticksBoost_INCLUDE_DIRS}  
    ${Assimp_INCLUDE_DIRS}  
    ${GGeo_INCLUDE_DIRS}  
    ${GLM_INCLUDE_DIRS}  
    ${NPY_INCLUDE_DIRS}  
    ${OpticksCore_INCLUDE_DIRS}  
)

add_definitions(
    ${OpticksBoost_DEFINITIONS}  
    ${Assimp_DEFINITIONS} 
    ${GGeo_DEFINITIONS} 
    ${GLM_DEFINITIONS} 
    ${NPY_DEFINITIONS} 
    ${OpticksCore_DEFINITIONS} 
)

set(LIBRARIES
    ${OpticksBoost_LIBRARIES} 
    ${Assimp_LIBRARIES} 
    ${GGeo_LIBRARIES} 
    ${NPY_LIBRARIES} 
    ${OpticksCore_LIBRARIES} 

    ssl
    crypto
)

set(SOURCES
    AssimpGeometry.cc
    AssimpCommon.cc
    AssimpNode.cc
    AssimpTree.cc
    AssimpRegistry.cc
    AssimpGGeo.cc
    AssimpSelection.cc
)

set(HEADERS
    AssimpCommon.hh
    AssimpGGeo.hh
    AssimpGeometry.hh
    AssimpNode.hh
    AssimpRegistry.hh
    AssimpSelection.hh
    AssimpTree.hh
)

add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} ${LIBRARIES})
install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})
add_subdirectory(tests EXCLUDE_FROM_ALL)

