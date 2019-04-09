cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OK)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)  

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()
 
find_package(OpticksGL REQUIRED CONFIG)

set(SOURCES

    OK_LOG.cc
    OKMgr.cc 
    OKPropagator.cc 
)
set(HEADERS

    OK_LOG.hh
    OK_API_EXPORT.hh
    OK_HEAD.hh
    OK_TAIL.hh

    OKMgr.hh
    OKPropagator.hh
)

add_library(${name} SHARED  ${SOURCES})

target_link_libraries( ${name} PUBLIC Opticks::OpticksGL )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
target_compile_definitions( ${name} PUBLIC OPTICKS_OPTIX OPTICKS_OK ) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


