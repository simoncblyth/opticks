cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpticksGeo)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

find_package(OpticksCore REQUIRED CONFIG)
find_package(OpenMeshRap REQUIRED CONFIG)

set(SOURCES

    OKGEO_LOG.cc
    OpticksAim.cc
    OpticksHub.cc
    OpticksGen.cc
    OpticksGun.cc
    OpticksIdx.cc

)

set(FORMER_SOURCES
    OpticksGeometry.cc
)

set(HEADERS

    OKGEO_LOG.hh
    OKGEO_API_EXPORT.hh
    OKGEO_HEAD.hh
    OKGEO_TAIL.hh

    OpticksHub.hh
    OpticksGen.hh
    OpticksGun.hh
    OpticksIdx.hh
)

set(FORMER_HEADERS
    OpticksGeometry.hh
)


add_library( ${name}  SHARED ${SOURCES})
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
target_compile_definitions( ${name} PUBLIC OPTICKS_OKGEO )
target_link_libraries( ${name} PUBLIC  Opticks::OpticksCore Opticks::OpenMeshRap )

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)


