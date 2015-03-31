cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name GLFWTest)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-switch-enum" )


########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OGLRap REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${OGLRap_INCLUDE_DIRS} 
)

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${OGLRap_DEFINITIONS}
)


add_executable( GLFWContext glfw_context.cc )
add_executable( ${name} glfw_test.cc app.cc ) 
add_executable( Simple simple.cc )

target_link_libraries( GLFWContext 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}
)
target_link_libraries( ${name} 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}
)
target_link_libraries( Simple 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}
)

install(TARGETS GLFWContext DESTINATION bin)
install(TARGETS ${name} DESTINATION bin)
install(TARGETS Simple DESTINATION bin)




