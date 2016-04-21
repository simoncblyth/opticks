#
# OPTICKS_PREFIX 
#     location beneath which ALL opticks packages are installed
#     and referenced from the FindX.cmake for cross usage
#
# OPTICKS_EXTERNAL_PREFIX 
#     location beneath which some opticks external packages
#     are installed, system type packages may be elsewhere
#     as specified by the FindX.cmake

message("${name}")

set(OPTICKS_PREFIX "$ENV{LOCAL_BASE}/env")
#set(OPTICKS_PREFIX "$ENV{LOCAL_BASE}/opticks")

set(OPTICKS_EXTERNAL_PREFIX "$ENV{LOCAL_BASE}/env")


OPTION(WITH_NPYSERVER  "using the numpyserver." OFF)
OPTION(WITH_OPTIX      "using OPTIX." OFF)


# /usr/local/env/chroma_env/src/root-v5.34.14/cmake/modules/RootBuildOptions.cmake
if (APPLE)
   # use, i.e. don't skip the full RPATH for the build tree
   set(CMAKE_SKIP_BUILD_RPATH  FALSE)
   # when building, don't use the install RPATH already (but later on when installing)
   set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
   # add the automatically determined parts of the RPATH
   # which point to directories outside the build tree to the install RPATH
   set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
   set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 
   # http://www.kitware.com/blog/home/post/510
   # enable @rpath in the install name for any shared library being built
   # note: it is planned that a future version of CMake will enable this by default
   set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)

