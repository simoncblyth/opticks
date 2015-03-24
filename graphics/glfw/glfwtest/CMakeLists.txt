cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name GLFWTest)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


find_package(GLFW REQUIRED)

include(EnvBuildOptions)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLFW_INCLUDE_DIRS} 
)

add_definitions(
   ${GLFW_DEFINITIONS}
)


add_executable( ${name} test.cc )

target_link_libraries( ${name} 
     ${GLFW_LIBRARIES}
)

install(TARGETS ${name} DESTINATION bin)




