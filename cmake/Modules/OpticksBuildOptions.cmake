#[=[
OpticksBuildOptions.cmake
============================

During integrated builds the project name is the top level 
one "Opticks" so the "name" variable is used here as
that works the same in integrated and proj-by-proj building.

OKConf package is found prior to setting up compilation flags in order 
to get the COMPUTE_CAPABILITY which is needed to define the CUDA_NVCC_FLAGS
see $(opticks-prefix)/lib/cmake/okconf/okconf-config.cmake

RPATH setup https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling


#]=]


#message(STATUS "OpticksBuildOptions.cmake Configuring ${name}")
message(STATUS "Configuring ${name}")

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(STATUS " CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR} ")
   message(STATUS " CMAKE_BINARY_DIR : ${CMAKE_BINARY_DIR} ")
   message(FATAL_ERROR "in-source build detected : DONT DO THAT")
endif()


include(CTest)
#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)
include(BCMSetupVersion)  # not yet used in anger, see examples/UseGLM
include(EchoTarget)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)



if(NOT ${name} STREQUAL "OKConf" AND NOT ${name} STREQUAL "OKConfTest")
  find_package(OKConf     REQUIRED CONFIG)   
endif()

include(OpticksCompilationFlags)   


