cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OpenMeshTest)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

########### using env/cmake/Modules/*.cmake #####

find_package(Boost REQUIRED)
find_package(GLM    REQUIRED)
find_package(NPY REQUIRED)
find_package(OpenMesh REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS} 
   ${OpenMesh_INCLUDE_DIRS} 
)

add_definitions(
   ${OpenMesh_DEFINITIONS}
   ${GLM_DEFINITIONS}
   ${NPY_DEFINITIONS}
   ${Boost_DEFINITIONS}
)

add_executable( ${name} 
     main.cc 
     MWrap.cc
) 

target_link_libraries( ${name} 
    ${OpenMesh_LIBRARIES}
    ${GLM_LIBRARIES}
    ${NPY_LIBRARIES}
    ${Boost_LIBRARIES}
)

install(TARGETS ${name} DESTINATION bin)


