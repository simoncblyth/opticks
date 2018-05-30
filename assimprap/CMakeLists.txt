cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name AssimpRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

find_package(OpticksAssimp REQUIRED MODULE)  ## external imported with cmake/Modules/FindOpticksAssimp.cmake
find_package(GGeo          REQUIRED CONFIG)


set(SOURCES

    ASIRAP_LOG.cc

    AssimpCommon.cc
    AssimpRegistry.cc
    AssimpSelection.cc
    AssimpTree.cc
    AssimpNode.cc

    AssimpGGeo.cc
    AssimpImporter.cc
)

set(HEADERS

    ASIRAP_LOG.hh
    ASIRAP_API_EXPORT.hh
    ASIRAP_HEAD.hh
    ASIRAP_TAIL.hh


    AssimpCommon.hh
    AssimpGGeo.hh
    AssimpImporter.hh
    AssimpNode.hh
    AssimpRegistry.hh
    AssimpSelection.hh
    AssimpTree.hh
)

add_library(${name} SHARED ${SOURCES})
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
target_compile_definitions( ${name} PUBLIC OPTICKS_ASIRAP )
target_link_libraries(${name} PUBLIC Opticks::OpticksAssimp Opticks::GGeo )

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

