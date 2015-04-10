cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name GLFWTest)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)


########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OGLRap REQUIRED)
find_package(Boost REQUIRED)

find_package(ZMQ REQUIRED)    # can cmake handle pkg depending on other pkgs, to make this simpler ?
find_package(AsioZMQ REQUIRED)
find_package(NumpyServer REQUIRED)
find_package(GLM  REQUIRED)
find_package(Cfg REQUIRED)


## c++11 forced by AsioZMQ
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif ()


#message(NumpyServer_INCLUDE_DIRS:${NumpyServer_INCLUDE_DIRS})

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${OGLRap_INCLUDE_DIRS} 
   ${Boost_INCLUDE_DIRS} 
   ${ZMQ_INCLUDE_DIRS} 
   ${AsioZMQ_INCLUDE_DIRS} 
   ${NumpyServer_INCLUDE_DIRS} 
   ${Cfg_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
)

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${OGLRap_DEFINITIONS}
)


add_executable( GLFWContext glfw_context.cc )
add_executable( Simple simple.cc )
add_executable( ${name} main.cc ) 

target_link_libraries( GLFWContext 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}
)
target_link_libraries( Simple 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}
)
target_link_libraries( ${name} 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}
      ${Boost_LIBRARIES}
      ${ZMQ_LIBRARIES}
      ${Cfg_LIBRARIES}
      ${NumpyServer_LIBRARIES}
)

install(TARGETS GLFWContext DESTINATION bin)
install(TARGETS Simple DESTINATION bin)
install(TARGETS ${name} DESTINATION bin)




