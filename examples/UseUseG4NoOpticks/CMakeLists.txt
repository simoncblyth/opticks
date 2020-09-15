cmake_minimum_required(VERSION 3.5)
set(name UseUseG4NoOpticks)
project(${name} VERSION 0.1.0)

# dont include OpticksBuildOptions : as "NoOpticks"
#include(OpticksBuildOptions)

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")  # override the GNUInstallDirs default of "include"

find_package(BCM CONFIG)
if(NOT BCM_FOUND)
   message(STATUS "CMAKE_MODULE_PATH:${CMAKE_MODULE_PATH}")
   message(STATUS "\$ENV{CMAKE_PREFIX_PATH}:$ENV{CMAKE_PREFIX_PATH}   expect to find BCM in one of these prefix dirs")
   message(FATAL_ERROR "ABORT " )
endif()

include(BCMDeploy)


find_package(UseG4NoOpticks 0.1.0 REQUIRED CONFIG)

add_executable(${name} ${name}.cc)

target_link_libraries( ${name} PUBLIC  Opticks::UseG4NoOpticks )   
# NB finding Geant4 happens automatically thanks to BCM and FindG4.cmake

install(TARGETS ${name} DESTINATION lib)



