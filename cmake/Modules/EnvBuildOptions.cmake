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

message("${name}")

set(OPTICKS_PREFIX "$ENV{LOCAL_BASE}/opticks")
set(OPTICKS_HOME   "$ENV{ENV_HOME}")

set(BUILD_SHARED_LIBS ON)

OPTION(WITH_NPYSERVER  "using the numpyserver." OFF)
OPTION(WITH_OPTIX      "using OPTIX." OFF)


# The _SOURCE_DIR are auto set for the superbuild
# but presumably not for individual buils. 
# Can see in cache::
#
#     simon:build blyth$ grep _SOURCE_DIR CMakeCache.txt
#
#
#set(AssimpRap_SOURCE_DIR   "${OPTICKS_HOME}/graphics/assimprap")
#set(BRegex_SOURCE_DIR      "${OPTICKS_HOME}/boost/bregex")
#set(CUDARap_SOURCE_DIR     "${OPTICKS_HOME}/cuda/cudarap")
#set(BCfg_SOURCE_DIR        "${OPTICKS_HOME}/boost/bpo/bcfg")
#set(GGeo_SOURCE_DIR        "${OPTICKS_HOME}/optix/ggeo")
#set(NPY_SOURCE_DIR         "${OPTICKS_HOME}/numerics/npy")
#set(NumpyServer_SOURCE_DIR "${OPTICKS_HOME}/boost/basio/numpyserver")
#set(OGLRap_SOURCE_DIR      "${OPTICKS_HOME}/graphics/oglrap")
#set(OpenMeshRap_SOURCE_DIR "${OPTICKS_HOME}/graphics/openmeshrap")
#set(OptiXRap_SOURCE_DIR    "${OPTICKS_HOME}/graphics/optixrap")
#set(OptiXThrust_SOURCE_DIR "${OPTICKS_HOME}/optix/optixthrust")
#set(Opticks_SOURCE_DIR     "${OPTICKS_HOME}/opticks")
#set(OpticksGL_SOURCE_DIR   "${OPTICKS_HOME}/opticksgl")
#set(OpticksOp_SOURCE_DIR   "${OPTICKS_HOME}/opticksop")
#set(PPM_SOURCE_DIR         "${OPTICKS_HOME}/graphics/ppm")
#set(ThrustRap_SOURCE_DIR   "${OPTICKS_HOME}/numerics/thrustrap")



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




