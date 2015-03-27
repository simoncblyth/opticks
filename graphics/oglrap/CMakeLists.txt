cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OGLRap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

########### using env/cmake/Modules/*.cmake #####
find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLM  REQUIRED)
find_package(GGeo REQUIRED)  # bit reluctant on this 

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${GGeo_INCLUDE_DIRS} 
)

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${GGeo_DEFINITIONS}
)

add_library( ${name}  SHARED
                Scene.cc
                Camera.cc
                View.cc
                Shader.cc
                Demo.cc
                VertexAttribute.cc
                Common.cc
)

add_executable( ${name}Test ${name}Test.cc )

target_link_libraries( ${name} 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${GGeo_LIBRARIES}
)
target_link_libraries( ${name}Test
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${GGeo_LIBRARIES}
      ${name}
)

install(TARGETS ${name}Test DESTINATION bin)
install(TARGETS ${name} DESTINATION lib)


SET(hdrs 
Scene.hh
Camera.hh
View.hh
Shader.hh
Demo.hh
VertexAttribute.hh
)

install(FILES ${hdrs} DESTINATION include)

