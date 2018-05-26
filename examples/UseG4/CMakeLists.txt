cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseG4)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(G4 MODULE REQUIRED)   

#echo_pfx_vars(G4 "FOUND;DIR;VERSION;VERSION_INTEGER;INCLUDE_DIR;INCLUDE_DIRS;LIBRARIES;DEFINITIONS")
echo_target(Opticks::G4 "INTERFACE_LINK_LIBRARIES;INTERFACE_INCLUDE_DIRECTORIES;INTERFACE_COMPILE_DEFINITIONS" )
#echo_target_twolevel(Opticks::G4 "INTERFACE_LINK_LIBRARIES" "IMPORTED_LOCATION")


set(SOURCES ${name}.cc)
set(HEADERS ${name}.hh)

add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} PUBLIC Opticks::G4)
#install(TARGETS ${name}  DESTINATION ${CMAKE_INSTALL_LIBDIR})   # bcm_deploy does this : no need to repeat
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)



