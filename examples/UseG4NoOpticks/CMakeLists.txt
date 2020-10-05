cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseG4NoOpticks)
project(${name} VERSION 0.1.0)

#[=[
The below preamble prior to find_package(G4.. ) 
usually included from OpticksBuildOptions.cmake but are here trying to be standalone.
#]=]

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")  # override the GNUInstallDirs default of "include"

find_package(BCM CONFIG)
if(NOT BCM_FOUND)
    message(STATUS "CMAKE_MODULE_PATH:${CMAKE_MODULE_PATH}")
    message(STATUS "\$ENV{CMAKE_PREFIX_PATH}:$ENV{CMAKE_PREFIX_PATH} ")
    message(FATAL_ERROR "ABORT : failed to find_package BCM : check envvar CMAKE_PREFIX_PATH " )
endif()
include(BCMDeploy)


set(G4_VERBOSE ON) 
find_package(G4 MODULE REQUIRED)   
message(STATUS "G4_DIRDIR : ${G4_DIRDIR} ")

set(SOURCES ${name}.cc)
set(HEADERS ${name}.hh)

add_library(${name} SHARED ${SOURCES})

#[=[
# Ancient variable-centric style of CMake 
include_directories( ${Geant4_INCLUDE_DIRS} )
target_link_libraries(${name} PUBLIC ${Geant4_LIBRARIES} )
#]=]

# target-centric style of CMake, which allows automatic handling of transitive dependencies thanks to BCM 
target_link_libraries(${name} PUBLIC Opticks::G4)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)


