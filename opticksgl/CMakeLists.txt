cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpticksGL)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


find_package(OGLRap REQUIRED CONFIG)
find_package(OKOP REQUIRED CONFIG)

# NB only for classes with OGLRap "OpenGL" dependency, for others use opticksop- not okgl-
set(SOURCES
     
    OKGL_LOG.cc

    OFrame.cc
    ORenderer.cc
    OAxisTest.cc

    OKGLTracer.cc
)
set(HEADERS

    OKGL_LOG.hh
    OKGL_API_EXPORT.hh
    OKGL_HEAD.hh
    OKGL_TAIL.hh

    OFrame.hh
    ORenderer.hh
    OAxisTest.hh

    OKGLTracer.hh
)

add_library( ${name}  SHARED ${SOURCES})
target_link_libraries( ${name} PUBLIC 
    Opticks::OGLRap
    Opticks::OKOP
)
target_compile_definitions( ${name} PUBLIC OPTICKS_OKGL )
target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)

