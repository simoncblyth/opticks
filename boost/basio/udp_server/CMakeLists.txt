cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name UDPServer)
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

add_executable( ${name} main.cpp ) 


target_link_libraries( ${name} 
      ${Boost_LIBRARIES}
)

install(TARGETS ${name} DESTINATION bin)

