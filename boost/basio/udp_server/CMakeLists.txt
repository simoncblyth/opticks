cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name UDPServer)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)


if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif ()




find_package(Boost REQUIRED)
find_package(ZMQ REQUIRED)
find_package(AsioZMQ REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${ZMQ_INCLUDE_DIRS} 
   ${AsioZMQ_INCLUDE_DIRS} 
)


add_executable( ${name} main.cpp ) 

target_link_libraries( ${name} 
      ${Boost_LIBRARIES}
      ${ZMQ_LIBRARIES} 
)

install(TARGETS ${name} DESTINATION bin)

