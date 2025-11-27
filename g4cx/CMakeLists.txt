cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(name G4CX)
set(desc "Top Level")

option(BUILD_WITH_CUDA "${name} Build with CUDA support" ON)  # default:ON/OFF
#[=[
After changing CUDA ON->OFF or OFF->ON need to reconf::

    om-clean
    om-conf
    om

#]=]

set(LANGUAGES CXX)
if(BUILD_WITH_CUDA)
    list(APPEND LANGUAGES CUDA)
endif()

project(${name} VERSION 0.1.0 LANGUAGES ${LANGUAGES})
include(OpticksBuildOptions NO_POLICY_SCOPE)

#[=[
G4CX
=====


#]=]

find_package(U4   REQUIRED CONFIG)
find_package(CSG  REQUIRED CONFIG)

if(BUILD_WITH_CUDA)
   find_package(CSGOptiX REQUIRED CONFIG)
   message(STATUS "${CMAKE_CURRENT_LIST_FILE} : CSGOptiX_FOUND     : ${CSGOptiX_FOUND}")
   message(STATUS "${CMAKE_CURRENT_LIST_FILE} : CSGOptiX_VERSION   : ${CSGOptiX_VERSION}")
else()
   message(STATUS "${CMAKE_CURRENT_LIST_FILE} : NOT-BUILD_WITH_CUDA so no CSGOptiX")
endif()

message(STATUS "${CMAKE_CURRENT_LIST_FILE} : Custom4_FOUND      : ${Custom4_FOUND}")
message(STATUS "${CMAKE_CURRENT_LIST_FILE} : Custom4_VERSION    : ${Custom4_VERSION}")


set(SOURCES
    G4CX_LOG.cc
    G4CXOpticks.cc
)

set(HEADERS
    G4CX_LOG.hh
    G4CX_API_EXPORT.hh
    G4CXOpticks.hh
)

add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

target_link_libraries( ${name} PUBLIC Opticks::CSG)
target_link_libraries( ${name} PUBLIC Opticks::U4)
if(BUILD_WITH_CUDA)
   target_compile_definitions( ${name} PUBLIC WITH_CUDA )
   target_link_libraries(      ${name} PUBLIC Opticks::CSGOptiX)
endif()

target_include_directories( ${name} PUBLIC
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_definitions( ${name} PUBLIC OPTICKS_G4CX )
target_compile_definitions( ${name} PUBLIC G4USE_STD11 )

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


