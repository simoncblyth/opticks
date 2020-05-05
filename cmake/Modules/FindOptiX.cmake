# // cp /Developer/OptiX_501/SDK/CMake/FindOptiX.cmake .
#
# Copyright (c) 2017 NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and proprietary
# rights in and to this software, related documentation and any modifications thereto.
# Any use, reproduction, disclosure or distribution of this software and related
# documentation without an express license agreement from NVIDIA Corporation is strictly
# prohibited.
#
# TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
# AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
# INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
# SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
# LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
# BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
# INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGES.
#

# Locate the OptiX distribution.  Search relative to the SDK first, then look in the system.

if(OptiX_VERBOSE)
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OptiX_VERBOSE     : ${OptiX_VERBOSE} ")
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OptiX_INSTALL_DIR : ${OptiX_INSTALL_DIR} ")
endif()

# Our initial guess will be within the SDK.
#set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../" CACHE PATH "Path to OptiX installed location.")

# The distribution contains both 32 and 64 bit libraries.  Adjust the library
# search path based on the bit-ness of the build.  (i.e. 64: bin64, lib64; 32:
# bin, lib).  Note that on Mac, the OptiX library is a universal binary, so we
# only need to look in lib and not lib64 for 64 bit builds.
if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT APPLE)
  set(bit_dest "64")
else()
  set(bit_dest "")
endif()

macro(OPTIX_find_api_library name version)
  find_library(${name}_LIBRARY
    NAMES ${name}.${version} ${name}
    PATHS "${OptiX_INSTALL_DIR}/lib${bit_dest}"
    NO_DEFAULT_PATH
    )
  find_library(${name}_LIBRARY
    NAMES ${name}.${version} ${name}
    )
  if(WIN32)
    find_file(${name}_DLL
      NAMES ${name}.${version}.dll
      PATHS "${OptiX_INSTALL_DIR}/bin${bit_dest}"
      NO_DEFAULT_PATH
      )
    find_file(${name}_DLL
      NAMES ${name}.${version}.dll
      )
  endif()
endmacro()

OPTIX_find_api_library(optix 1)
OPTIX_find_api_library(optixu 1)
OPTIX_find_api_library(optix_prime 1)


# Include
find_path(OptiX_INCLUDE
  NAMES optix.h
  PATHS "${OptiX_INSTALL_DIR}/include"
  NO_DEFAULT_PATH
  )


find_path(OptiX_INCLUDE
  NAMES optix.h
  )


# Check to make sure we found what we were looking for
function(OptiX_report_error error_message required)
  if(OptiX_FIND_REQUIRED AND required)
    message(FATAL_ERROR "${error_message}")
  else()
    if(NOT OptiX_FIND_QUIETLY)
      message(STATUS "${error_message}")
    endif(NOT OptiX_FIND_QUIETLY)
  endif()
endfunction()

set(_hint "Try adding cmake argument: -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir)  (all packages downstream from OptiX need this)")


if(NOT optix_LIBRARY)
  OptiX_report_error("optix library not found. ${_hint} " TRUE)
endif()
if(NOT OptiX_INCLUDE)
  OptiX_report_error("OptiX headers (optix.h and friends) not found. ${_hint}" TRUE)
endif()
if(NOT optix_prime_LIBRARY)
  OptiX_report_error("optix Prime library not found. ${_hint}" FALSE)
endif()

# Macro for setting up dummy targets
function(OptiX_add_imported_library name lib_location dll_lib dependent_libs)
  set(CMAKE_IMPORT_FILE_VERSION 1)

  # Create imported target
  add_library(${name} SHARED IMPORTED)

  # Import target "optix" for configuration "Debug"
  if(WIN32)
    set_target_properties(${name} PROPERTIES
      IMPORTED_IMPLIB "${lib_location}"
      #IMPORTED_LINK_INTERFACE_LIBRARIES "glu32;opengl32"
      IMPORTED_LOCATION "${dll_lib}"
      IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
      INTERFACE_IMPORTED_LOCATION "${lib_location}"             # workaround 3.14 whitelisting
      INTERFACE_IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
      )
  elseif(UNIX)
    set_target_properties(${name} PROPERTIES
      #IMPORTED_LINK_INTERFACE_LIBRARIES "glu32;opengl32"
      IMPORTED_LOCATION "${lib_location}"
      # We don't have versioned filenames for now, and it may not even matter.
      #IMPORTED_SONAME "${optix_soname}"
      IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
      INTERFACE_IMPORTED_LOCATION "${lib_location}"             # workaround 3.14 whitelisting
      INTERFACE_IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
      )
  else()
    # Unknown system, but at least try and provide the minimum required
    # information.
    set_target_properties(${name} PROPERTIES
      IMPORTED_LOCATION "${lib_location}"
      IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
      INTERFACE_IMPORTED_LOCATION "${lib_location}"             # workaround 3.14 whitelisting
      INTERFACE_IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
      )
  endif()

  # Commands beyond this point should not need to know the version.
  set(CMAKE_IMPORT_FILE_VERSION)
endfunction()

# Sets up a dummy target
OptiX_add_imported_library(optix "${optix_LIBRARY}" "${optix_DLL}" "${OPENGL_LIBRARIES}")
OptiX_add_imported_library(optixu   "${optixu_LIBRARY}"   "${optixu_DLL}"   "")
OptiX_add_imported_library(optix_prime "${optix_prime_LIBRARY}"  "${optix_prime_DLL}"  "")


set(OptiX_targets)

if(TARGET optix)
list(APPEND OptiX_targets optix)
endif()

if(TARGET optixu)
list(APPEND OptiX_targets optixu)
endif()

if(TARGET optix_prime)
list(APPEND OptiX_targets optix_prime)
endif()




macro(OptiX_check_same_path libA libB)
  if(_optix_path_to_${libA})
    if(NOT _optix_path_to_${libA} STREQUAL _optix_path_to_${libB})
      # ${libA} and ${libB} are in different paths.  Make sure there isn't a ${libA} next
      # to the ${libB}.
      get_filename_component(_optix_name_of_${libA} "${${libA}_LIBRARY}" NAME)
      if(EXISTS "${_optix_path_to_${libB}}/${_optix_name_of_${libA}}")
        message(WARNING " ${libA} library found next to ${libB} library that is not being used.  Due to the way we are using rpath, the copy of ${libA} next to ${libB} will be used during loading instead of the one you intended.  Consider putting the libraries in the same directory or moving ${_optix_path_to_${libB}}/${_optix_name_of_${libA} out of the way.")
      endif()
    endif()
    set( _${libA}_rpath "-Wl,-rpath,${_optix_path_to_${libA}}" )
  endif()
endmacro()

# Since liboptix.1.dylib is built with an install name of @rpath, we need to
# compile our samples with the rpath set to where optix exists.
if(APPLE)
  get_filename_component(_optix_path_to_optix "${optix_LIBRARY}" PATH)
  if(_optix_path_to_optix)
    set( _optix_rpath "-Wl,-rpath,${_optix_path_to_optix}" )
  endif()
  get_filename_component(_optix_path_to_optixu "${optixu_LIBRARY}" PATH)
  OptiX_check_same_path(optixu optix)
  get_filename_component(_optix_path_to_optix_prime "${optix_prime_LIBRARY}" PATH)
  OptiX_check_same_path(optix_prime optix)
  OptiX_check_same_path(optix_prime optixu)

  set( optix_rpath ${_optix_rpath} ${_optixu_rpath} ${_optix_prime_rpath} )
  if(optix_rpath)
      list(REMOVE_DUPLICATES optix_rpath)
  endif() 
endif()




## Opticks additions

set(OptiX_MODULE  "${CMAKE_CURRENT_LIST_FILE}")  
## TODO: Discover general way that finders of CMake modules can introspect the path of the one that was found ?

if(optix_LIBRARY AND optixu_LIBRARY AND OptiX_INCLUDE)
   set(OptiX_FOUND "YES")
   get_filename_component(OptiX_LIBDIR ${optix_LIBRARY} DIRECTORY)
else()
   set(OptiX_FOUND "NO")
endif()

set(OptiX_VERSION_INTEGER 0)
if(OptiX_FOUND)
   file(READ "${OptiX_INCLUDE}/optix.h" _contents)
   #message(STATUS "FindOptiX.cmake:_contents : ${_contents} ")
   string(REGEX REPLACE "\n" ";" _contents "${_contents}")
   foreach(_line ${_contents})
        if (_line MATCHES "#define OPTIX_VERSION ([0-9]+) ")
            set(OptiX_VERSION_INTEGER ${CMAKE_MATCH_1} )
            #message(STATUS "FindOptiX.cmake._line ${_line} ===> ${CMAKE_MATCH_1} ") 
        endif()
   endforeach()
endif()

#set(OptiX_VERBOSE NO)
if(OptiX_VERBOSE)
   message(STATUS "FindOptiX.cmake.OptiX_MODULE          : ${OptiX_MODULE}")
   message(STATUS "FindOptiX.cmake.OptiX_FOUND           : ${OptiX_FOUND}")
   message(STATUS "FindOptiX.cmake.OptiX_VERSION_INTEGER : ${OptiX_VERSION_INTEGER}")
   message(STATUS "FindOptiX.cmake.OptiX_INCLUDE         : ${OptiX_INCLUDE}")
   message(STATUS "FindOptiX.cmake.OptiX_LIBDIR          : ${OptiX_LIBDIR}")
   message(STATUS "FindOptiX.cmake.optix_LIBRARY         : ${optix_LIBRARY}")
   message(STATUS "FindOptiX.cmake.optixu_LIBRARY        : ${optixu_LIBRARY}")
   message(STATUS "FindOptiX.cmake.optix_prime_LIBRARY   : ${optix_prime_LIBRARY}")
endif()

if(OptiX_FOUND)
   add_library(Opticks::OptiX INTERFACE IMPORTED) 
   set_target_properties(Opticks::OptiX PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OptiX_INCLUDE}"
        INTERFACE_LINK_LIBRARIES "optix;optixu;optix_prime"
        INTERFACE_PKG_CONFIG_NAME "OptiX"
   )

   list(APPEND OptiX_targets OptiX)

endif()


