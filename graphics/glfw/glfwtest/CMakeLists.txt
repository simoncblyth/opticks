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

########## using pkg-config approach and GLFW3.pc file  #####
#
#find_package(PkgConfig)
#pkg_check_modules(PC_GLFW QUIET glfw3)  # looking for glfw3.pc case sensitive
#
#find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h
#          HINTS ${PC_GLFW_INCLUDEDIR} ${PC_GLFW_INCLUDE_DIRS}
#          )
#message(GLFW_INCLUDE_DIR:${GLFW_INCLUDE_DIR})
#
#find_library(GLFW_LIBRARY NAMES glfw
#             HINTS ${PC_GLFW_LIBDIR} ${PC_GLFW_LIBRARY_DIRS} )
#message(GLFW_LIBRARY:${GLFW_LIBRARY})
#
#set(GLFW_DEFINITIONS ${PC_GLFW_CFLAGS_OTHER})
#message(GLFW_DEFINITIONS:${GLFW_DEFINITIONS})
#
#set(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR} )
#
#if(APPLE)
#    find_library( Cocoa_LIBRARY NAMES Cocoa )
#    find_library( OpenGL_LIBRARY NAMES OpenGL )
#    find_library( IOKit_LIBRARY NAMES IOKit )
#    find_library( CoreFoundation_LIBRARY NAMES CoreFoundation )
#    find_library( CoreVideo_LIBRARY NAMES CoreVideo )
#
#    set( GLFW_LIBRARIES 
#               ${GLFW_LIBRARY} 
#               ${Cocoa_LIBRARY}
#               ${OpenGL_LIBRARY}
#               ${IOKit_LIBRARY}
#               ${CoreFoundation_LIBRARY} 
#               ${CoreVideo_LIBRARY} )
#endif(APPLE)
##################################################


include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
)

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
)


add_executable( GLFWContext glfw_context.cc )
add_executable( ${name} 
                glfw_test.cc 
                Geometry.cc
                Array.cc
                VertexAttribute.cc
                VertexBuffer.cc
                Shader.cc
)
add_executable( Simple simple.cc )

target_link_libraries( GLFWContext 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
)
target_link_libraries( ${name} 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
)
target_link_libraries( Simple 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
)

install(TARGETS GLFWContext DESTINATION bin)
install(TARGETS ${name} DESTINATION bin)
install(TARGETS Simple DESTINATION bin)




