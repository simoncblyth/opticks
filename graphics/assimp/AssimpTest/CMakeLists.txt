cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

#set(name AssimpTest)
set(name AssimpTestPP)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

find_package(Assimp REQUIRED)

include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Assimp_INCLUDE_DIRS}  
)


add_definitions(
    ${Assimp_DEFINITIONS} 
)


add_executable(${name} ${name}.cc )

target_link_libraries( ${name} 
                        ${Assimp_LIBRARIES} 
                      )

install(TARGETS ${name} DESTINATION bin)
