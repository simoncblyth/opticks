cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name GLFWTestTests)
project(${name})

add_executable( GLFWContext glfw_context.cc )
add_executable( Simple simple.cc )

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

install(TARGETS GLFWContext DESTINATION bin)
install(TARGETS Simple DESTINATION bin)


