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

message(STATUS "Configuring ${name}")
if(OBO_VERBOSE)
message(STATUS "OpticksBuildOptions.cmake Configuring ${name} [")
endif()


if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(STATUS " CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR} ")
   message(STATUS " CMAKE_BINARY_DIR : ${CMAKE_BINARY_DIR} ")
   message(FATAL_ERROR "in-source build detected : DONT DO THAT")
endif()


if(NOT OPTICKS_PREFIX)
    get_filename_component(OBO_MODULE_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
    get_filename_component(OBO_MODULE_DIRDIR ${OBO_MODULE_DIR} DIRECTORY)
    get_filename_component(OBO_MODULE_DIRDIRDIR ${OBO_MODULE_DIRDIR} DIRECTORY)
    set(OPTICKS_PREFIX ${OBO_MODULE_DIRDIRDIR})
    # this gives correct prefix when this module is included from installed tree
    # but when included from source tree it gives home
    # hence use -DOPTICKS_PREFIX=$(om-prefix) for cmake internal source builds
    # so that OPTICKS_PREFIX is always correct
endif()


# initialize a list into which targets found by cmake/Modules/FindXXX.cmake are appended
set(OPTICKS_TARGETS_FOUND)


include(CTest)
#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)
include(BCMSetupVersion)  # not yet used in anger, see examples/UseGLM
include(EchoTarget)
include(TopMetaTarget)

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
    if(CMAKE_INSTALL_PREFIX STREQUAL ${OPTICKS_PREFIX})
       set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64")
    else()
       message(STATUS " Below two strings differ : forced to use absolute RPATH ")
       message(STATUS " CMAKE_INSTALL_PREFIX : ${CMAKE_INSTALL_PREFIX} ")
       message(STATUS " OPTICKS_PREFIX       : ${OPTICKS_PREFIX} ")
       set(ABSOLUTE_INSTALL_RPATH
                     ${OPTICKS_PREFIX}/lib64  
                     ${OPTICKS_PREFIX}/externals/lib  
                     ${OPTICKS_PREFIX}/externals/lib64  
                     ${OPTICKS_PREFIX}/externals/OptiX/lib64  
          ) 
       set(CMAKE_INSTALL_RPATH  "${ABSOLUTE_INSTALL_RPATH}")
    endif()

elseif(APPLE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()





include(OpticksCXXFlags)   

if(OBO_VERBOSE)
message(STATUS "OpticksBuildOptions.cmake Configuring ${name} ]")
endif()


