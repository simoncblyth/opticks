#[=[
UseInstance
===========

Updated of old env- instance- example ~/env/graphics/opengl/instance, 
CMake modernization.

Hmm : but RPATH setup not workinh 

#]=]
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(name UseInstance)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(GLM         REQUIRED MODULE)
find_package(OpticksGLEW REQUIRED MODULE)
find_package(OpticksGLFW REQUIRED MODULE)   


set(SOURCES
     Prog.cc
     Frame.cc
     Buf.cc
     Renderer.cc
)

set(HEADERS
     Prog.hh
     Frame.hh
     Buf.hh
     Renderer.hh
)
 
add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

target_include_directories( ${name} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
) 

target_link_libraries( ${name} PUBLIC
    Opticks::GLM 
    Opticks::OpticksGLEW
    Opticks::OpticksGLFW
)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

