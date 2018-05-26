cmake_minimum_required(VERSION 3.5)
set(name UseUseG4)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


find_package(UseG4 0.1.0 REQUIRED)

echo_target_std(Opticks::UseG4)
echo_target(Opticks::G4 "INTERFACE_LINK_LIBRARIES;INTERFACE_COMPILE_DEFINITIONS;INTERFACE_INCLUDE_DIRECTORIES")
#echo_target_twolevel(Opticks::G4 "INTERFACE_LINK_LIBRARIES" "IMPORTED_LOCATION")

add_executable(${name} ${name}.cc)

target_link_libraries( ${name} PUBLIC  Opticks::UseG4 )

install(TARGETS ${name} DESTINATION lib)



