cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OKConf)
project(${name} VERSION 0.1.0)
set(FLAGS_VERBOSE ON)
include(OpticksBuildOptions)
set(FLAGS_VERBOSE OFF)

set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_FULL_LIBDIR})  

#[=[
OKConf
========

OKConf is unusual : it find_packages OptiX and G4, see::

    cmake/Modules/FindOptiX.cmake 
    cmake/Modules/FindG4.cmake 

Within those the location of the INCLUDE_DIR is used 
to access headers which are parsed to extract version integers. 
But the OptiX and G4 targets are not linked against.
The below default RPATH setup from OpticksBuildOptions of
does not work for OKConfTest ? But the above does::

   set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) 

#]=]


find_package(OpticksCUDA MODULE QUIET) 
find_package(OptiX       MODULE QUIET) 
find_package(G4          MODULE QUIET) 


# generate define header for hardcoding in the library 

set(OKCONF_CUDA_API_VERSION_INTEGER "${OpticksCUDA_API_VERSION}")
set(OKCONF_OPTIX_VERSION_INTEGER  "${OptiX_VERSION_INTEGER}")
set(OKCONF_OPTIX_INSTALL_DIR      "${OptiX_INSTALL_DIR}")
set(OKCONF_GEANT4_VERSION_INTEGER "${G4_VERSION_INTEGER}")

set(OKCONF_OPTICKS_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

set(OKCONF_COMPUTE_CAPABILITY_INTEGER "${COMPUTE_CAPABILITY}")
set(OKCONF_CUDA_NVCC_FLAGS     "${CUDA_NVCC_FLAGS}")
set(OKCONF_CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS}")


# TODO:get rid of the below : adopt the above names
set(OXRAP_OPTIX_VERSION   "${OptiX_VERSION_INTEGER}")
set(CFG4_G4VERSION_NUMBER "${G4_VERSION_INTEGER}")



set(OKConf_GENERATED_HEADER  "OKConf_Config.hh")

#configure_file( OKCONF_OpticksCMakeConfig.hh.in inc/${OKCONF_GENERATED_HEADER} )
configure_file( ${OKConf_GENERATED_HEADER}.in inc/${OKConf_GENERATED_HEADER} )

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

# hmm reluctant to make PUBLIC as just want to communicate to the tests 
# but for integrated it seems need to, unless just use ".." from the tests

target_include_directories( ${name} PRIVATE
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/inc>
)


add_subdirectory(tests)




 

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL TOPMATTER "
## OKConf generated TOPMATTER

set(OptiX_INSTALL_DIR ${OptiX_INSTALL_DIR})
set(COMPUTE_CAPABILITY ${COMPUTE_CAPABILITY})

if(OKConf_VERBOSE)
  message(STATUS \"\${CMAKE_CURRENT_LIST_FILE} : OKConf_VERBOSE     : \${OKConf_VERBOSE} \")
  message(STATUS \"\${CMAKE_CURRENT_LIST_FILE} : OptiX_INSTALL_DIR  : \${OptiX_INSTALL_DIR} \")
  message(STATUS \"\${CMAKE_CURRENT_LIST_FILE} : COMPUTE_CAPABILITY : \${COMPUTE_CAPABILITY} \")
endif()
" )


install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/inc/${OKConf_GENERATED_HEADER}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


