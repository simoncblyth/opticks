#
# OPTICKS_PREFIX 
#     location beneath which ALL opticks packages are installed
#     and referenced from the FindX.cmake for cross usage
#
# The distinction between what to consider external/internal
# (assuming you have the source) boils down to how often you want 
# to recompile. Once a package has solidified promoting it to be 
# an external allows to skip from everyday project rebuilding.
#

message(STATUS "Configuring ${name}")

#
# FindX.cmake use OPTICKS_PREFIX and _SOURCE_DIR
# set by CMake as visible in cache : grep _SOURCE_DIR CMakeCache.txt
#
set(OPTICKS_PREFIX "$ENV{LOCAL_BASE}/opticks")

set(OPTICKS_CUDA_VERSION 5.5)
set(OPTICKS_OPTIX_VERSION 3.5)

set(BUILD_SHARED_LIBS ON)

OPTION(WITH_NPYSERVER  "using the numpyserver." OFF)
OPTION(WITH_OPTIX      "using OPTIX." OFF)


#
# https://cmake.org/Wiki/CMake_RPATH_handling

if (APPLE)
   set(CMAKE_SKIP_BUILD_RPATH  FALSE)
   set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
   set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")  
   set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 
   # http://www.kitware.com/blog/home/post/510
   # enable @rpath in the install name for any shared library being built
   # note: it is planned that a future version of CMake will enable this by default
   set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)


if(UNIX AND NOT APPLE)
   set(CMAKE_SKIP_BUILD_RPATH  FALSE)
   set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
   set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

   # the RPATH to be used when installing, but only if it's not a system directory
   list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
   if("${isSystemDir}" STREQUAL "-1")
      set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
   endif("${isSystemDir}" STREQUAL "-1")

endif(UNIX AND NOT APPLE)




