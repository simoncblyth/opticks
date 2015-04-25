cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name bpt)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)
find_package(Boost REQUIRED)


include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
)

add_definitions(
   ${Boost_DEFINITIONS} 
)

add_executable(${name}xml settings.cpp)
add_executable(${name}ini bookmarks.cpp)


install(TARGETS ${name}xml DESTINATION bin)
install(TARGETS ${name}ini DESTINATION bin)




