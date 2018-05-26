#[=[
UseUseG4DAE : Consume functionality from UseG4DAE lib
========================================================

#]=]

cmake_minimum_required(VERSION 3.5)
set(name UseUseG4DAE)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(UseG4DAE 0.1.0 REQUIRED)

#echo_target_std(Opticks::UseG4DAE)
#echo_target(Opticks::G4DAE "INTERFACE_LINK_LIBRARIES;INTERFACE_COMPILE_DEFINITIONS;INTERFACE_INCLUDE_DIRECTORIES")

add_executable(${name} ${name}.cc)

target_link_libraries( ${name} PUBLIC  Opticks::UseG4DAE )

install(TARGETS ${name} DESTINATION lib)


