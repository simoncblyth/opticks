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

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
)

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
)

add_library( ${name}  SHARED
                Array.cc
                Scene.cc
                Shader.cc
                Buffer.cc
                Geometry.cc
                IGeometry.cc
                VertexAttribute.cc
)

add_executable( ${name}Test ${name}Test.cc )

target_link_libraries( ${name} 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
)
target_link_libraries( ${name}Test
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${name}
)

install(TARGETS ${name}Test DESTINATION bin)
install(TARGETS ${name} DESTINATION lib)


SET(hdrs 
Array.hh
Buffer.hh
Geometry.hh
IGeometry.hh
Scene.hh
Shader.hh
VertexAttribute.hh
)

install(FILES ${hdrs} DESTINATION include)

