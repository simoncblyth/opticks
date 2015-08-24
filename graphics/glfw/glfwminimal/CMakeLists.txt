cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name GLFWMinimal)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)


########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_library(OpenGL_LIBRARY NAMES GL )


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
)

set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
)

set(LIBRARIES
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OpenGL_LIBRARY}
)

include_directories( ${INCLUDE_DIRS} )

add_definitions( ${DEFINITIONS} )

add_executable( ${name} glfwminimal.cc ) 

target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION bin)

add_subdirectory(tests)

