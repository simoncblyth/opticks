cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
set(name OKConf)
project(${name} VERSION 0.1.0)

#include(EnvBuildOptions)
include(EnvCompilationFlags)

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")



set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_FULL_LIBDIR})  
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(BCM)
include(BCMDeploy)

## OKConf is unusual : it parses OptiX and G4 package headers 
## to find version integers and otherwise does not link to those packages.

find_package(OptiX MODULE QUIET) 
find_package(G4    MODULE QUIET) 

# generate define header for hardcoding in the library 
set(OKCONF_OPTIX_VERSION_INTEGER  "${OptiX_VERSION_INTEGER}")
set(OKCONF_GEANT4_VERSION_INTEGER "${G4_VERSION_INTEGER}")

# TODO:get rid of the below : adopt the above names
set(OXRAP_OPTIX_VERSION   "${OptiX_VERSION_INTEGER}")
set(CFG4_G4VERSION_NUMBER "${G4_VERSION_INTEGER}")

configure_file( OpticksCMakeConfig.hh.in inc/OpticksCMakeConfig.hh )

# generate opticks-config sh script into lib dir
#include(OpticksConfigureConfigScript)



set(SOURCES
    OKConf.cc
)

set(HEADERS
    OKConf.hh
    OKCONF_API_EXPORT.hh
)

add_library( ${name}  SHARED ${SOURCES})

target_include_directories( ${name} PUBLIC
   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/inc>
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
 
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/inc/OpticksCMakeConfig.hh  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

