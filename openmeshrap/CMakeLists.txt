cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpenMeshRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(GGeo   REQUIRED CONFIG)
find_package(OpticksCore REQUIRED CONFIG)

set(SOURCES
     MESHRAP_LOG.cc

     MMesh.cc
     MWrap.cc
     MTool.cc
     MFixer.cc
)
set(HEADERS

     MESHRAP_LOG.hh
     MESHRAP_API_EXPORT.hh
     MESHRAP_HEAD.hh
     MESHRAP_TAIL.hh

     MMesh.hh
     MWrap.hh
     MTool.hh
     MFixer.hh
)

add_library( ${name}  SHARED ${SOURCES})
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
target_compile_definitions( ${name} PUBLIC OPTICKS_MESHRAP )
target_link_libraries( ${name} PUBLIC  Opticks::OpticksCore Opticks::GGeo )

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

