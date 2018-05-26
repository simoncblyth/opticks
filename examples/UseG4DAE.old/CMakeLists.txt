cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseG4DAE)
project(${name} VERSION 0.1.0)

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)
    
include(EchoTarget)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(G4DAE_VERBOSE ON)
find_package(G4    MODULE REQUIRED)   
find_package(G4DAE MODULE REQUIRED)   
find_package(OpticksXercesC MODULE REQUIRED)   

#echo_pfx_vars(G4 "FOUND;DIR;VERSION;VERSION_INTEGER;INCLUDE_DIR;INCLUDE_DIRS;LIBRARIES;DEFINITIONS")
echo_target(Opticks::G4DAE "INTERFACE_LINK_LIBRARIES;INTERFACE_INCLUDE_DIRECTORIES;INTERFACE_COMPILE_DEFINITIONS" )
#echo_target_twolevel(Opticks::G4 "INTERFACE_LINK_LIBRARIES" "IMPORTED_LOCATION")

set(SOURCES ${name}.cc)
set(HEADERS ${name}.hh)

add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} 
   PRIVATE
       Opticks::G4
       Opticks::G4DAE
       Opticks::OpticksXercesC
)

target_include_directories(${name} PUBLIC
   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)


