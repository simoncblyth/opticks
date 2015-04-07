cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name CfgTest)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

## c++11 forced by AsioZMQ
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif ()


########### using env/cmake/Modules/*.cmake #####

find_package(Boost REQUIRED)
find_package(Cfg  REQUIRED)
find_package(GLM  REQUIRED)
find_package(OGLRap REQUIRED)


include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${Cfg_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${OGLRap_INCLUDE_DIRS} 
)

add_definitions(
   ${OGLRap_DEFINITIONS}
)

add_executable( ${name}
   main.cc 
) 

target_link_libraries( ${name}
      ${Boost_LIBRARIES}
      ${OGLRap_LIBRARIES}
      ${Cfg_LIBRARIES}
)


install(TARGETS ${name}    DESTINATION bin)



