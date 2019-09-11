#[=[
OpticksBuildOptions.cmake
============================

This is included into all Opticks subproject CMakeLists.txt
Formerly did a conditional find_package for OKConf here::

   if(NOT ${name} STREQUAL "OKConf" AND NOT ${name} STREQUAL "OKConfTest")
      find_package(OKConf     REQUIRED CONFIG)   
   endif()

But it is confusing to hide a package dependency like this, 
it is better to be explicit and have opticks-deps give a true picture 
of the dependencies. BCM is an exception as it is CMake level only 
infrastructure. 

Also formerly included OpticksCUDAFlags which defines CUDA_NVCC_FLAGS here, 
but that depends on the COMPUTE_CAPABILITY that is provided by OKConf 
so have moved that to the OKConf generated TOPMATTER, which generates:

*  $(opticks-prefix)/lib/cmake/okconf/okconf-config.cmake

RPATH setup docs 

* https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling

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




# macOS RPATH
# ------------
#
# CMAKE_INSTALL_RPATH_USE_LINK_PATH : adds the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
#
# see env-;otool-;otool-rpath  
#
# * https://blogs.oracle.com/dipol/dynamic-libraries,-rpath,-and-mac-os
#
#
# Linux RPATH 
# --------------------
#
# install RPATH is prefixed with $ORIGIN/.. to simplify deployment of Opticks binaries 
# users then only need to set PATH and the executables are able to find the libs relative to themselves  
# see notes/issues/packaging-opticks-and-externals-for-use-on-gpu-cluster.rst 
#
# to check the RPATH of a library or executable use chrpath on it, eg: chrpath $(which OKTest) 
#


if(UNIX AND NOT APPLE)
#set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64")
elseif(APPLE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()





include(OpticksCXXFlags)   


