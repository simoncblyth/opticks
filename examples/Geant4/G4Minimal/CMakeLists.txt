cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name G4Minimal)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(G4 REQUIRED MODULE)

set(SOURCES 
    ${name}.cc
    DetectorConstruction.cc
    RunAction.cc
)

add_executable(${name} ${SOURCES})

target_link_libraries( ${name} Opticks::G4 )


install(TARGETS ${name}  DESTINATION ${CMAKE_INSTALL_LIBDIR}) 



