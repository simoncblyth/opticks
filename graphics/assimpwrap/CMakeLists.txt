cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name AssimpWrap)
project(${name})


set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    "$ENV{ENV_HOME}/cmake/Modules"
) 

include(EnvBuildOptions)

set(BUILD_SHARED_LIBS ON)



find_package(Boost REQUIRED)
find_package(Assimp REQUIRED)
find_package(GGeo REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)
find_package(Opticks REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}  
    ${Assimp_INCLUDE_DIRS}  
    ${GGeo_INCLUDE_DIRS}  
    ${GLM_INCLUDE_DIRS}  
    ${NPY_INCLUDE_DIRS}  
    ${Opticks_INCLUDE_DIRS}  
)

add_definitions(
    ${Boost_DEFINITIONS}  
)

add_library(${name} SHARED 
    AssimpGeometry.cc
    AssimpCommon.cc
    AssimpNode.cc
    AssimpTree.cc
    AssimpRegistry.cc
    AssimpGGeo.cc
    AssimpSelection.cc
)

set(LIBRARIES
    ${Boost_LIBRARIES} 
    ${Assimp_LIBRARIES} 
    ${GGeo_LIBRARIES} 
    ${NPY_LIBRARIES} 
    ${Opticks_LIBRARIES} 
    ssl
    crypto
)

set(hdrs
    AssimpCommon.hh
    AssimpGGeo.hh
    AssimpGeometry.hh
    AssimpNode.hh
    AssimpRegistry.hh
    AssimpSelection.hh
    AssimpTree.hh
)


target_link_libraries(${name} ${LIBRARIES})

install(FILES  ${hdrs} DESTINATION include) 

install(TARGETS ${name} LIBRARY DESTINATION lib)

add_subdirectory(tests)


