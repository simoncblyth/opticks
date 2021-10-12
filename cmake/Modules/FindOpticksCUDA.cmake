
#[=[
FindOpticksCUDA.cmake
=======================

In order to consistently pickup a new CUDA version have observed 
that it is necessary to: 

1. have a "clean" PATH and LD_LIBRARY_PATH with only one instance of CUDA binaries 
   like nvcc and libraries within it. Check with::

        echo $PATH | tr ":" "\n"
        echo $LD_LIBRARY_PATH | tr ":" "\n"

   An easy way to do this is to do the environment setup in .bashrc OR .bash_profile
   and then logout and back in again.

2. om-clean
3. om-conf

The reason for this black magic is that FindCUDA.cmake which comes 
with CMake is being treated as a black box. However the primary 
way that FindCUDA operates is effectively from where nvcc is
found in the PATH.   


#]=]



if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    message(STATUS "FindOpticksCUDA.cmake:with-policy-CMP0077") 
    cmake_policy(SET CMP0077 OLD)
    #cmake_policy(SET CMP0077 NEW)
else()
    message(STATUS "FindOpticksCUDA.cmake:no-policy-CMP0077") 
endif()



#find_package(CUDA   REQUIRED MODULE) 
find_package(CUDA   MODULE) 

# actual finding done by module supplied with CMake such as:
# /opt/local/share/cmake-3.17/Modules/FindCUDA.cmake



if(CUDA_LIBRARIES AND CUDA_INCLUDE_DIRS AND CUDA_curand_LIBRARY)
  set(OpticksCUDA_FOUND "YES")
  set(OpticksCUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
  set(OpticksCUDA_LIBRARIES    ${CUDA_LIBRARIES}) 
  get_filename_component(OpticksCUDA_LIBDIR ${CUDA_curand_LIBRARY} DIRECTORY) 
else()
  set(OpticksCUDA_FOUND "NO")
endif()


set(CUDA_API_VERSION_INTEGER 0)
if(OpticksCUDA_FOUND)
   file(READ "${CUDA_INCLUDE_DIRS}/cuda.h" _contents)
   string(REGEX REPLACE "\n" ";" _contents "${_contents}")
   foreach(_line ${_contents})
       #if (_line MATCHES "^    #define __CUDA_API_VERSION ([0-9]+)") ## require 4 spaces to distinguish from another ancient API version 
       if (_line MATCHES "#define CUDA_VERSION ([0-9]+)") ## require 4 spaces to distinguish from another ancient API version 
            set(OpticksCUDA_API_VERSION ${CMAKE_MATCH_1} )
            #message(STATUS "FindOpticksCUDA.cmake:OpticksCUDA_API_VERSION:${OpticksCUDA_API_VERSION}") 
       endif()
   endforeach()

   # see notes/issues/helper_cuda.rst
   find_path(
       HELPER_CUDA_INCLUDE_DIR
       NAMES "helper_cuda.h"
       PATHS
           "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc"
           "${CUDA_TOOLKIT_ROOT_DIR}/cuda-samples/common/inc"
           "${CMAKE_CURRENT_LIST_DIR}/include/helper_cuda_fallback/${CUDA_VERSION}"
    )
    if(HELPER_CUDA_INCLUDE_DIR)
        set(OpticksHELPER_CUDA_FOUND "YES")
    else()
        set(OpticksHELPER_CUDA_FOUND "NO")
    endif()

endif()

if(OpticksCUDA_VERBOSE)

  message(STATUS "Use examples/UseOpticksCUDA/CMakeLists.txt for testing FindOpticksCUDA.cmake" )
  message(STATUS "  CUDA_TOOLKIT_ROOT_DIR   : ${CUDA_TOOLKIT_ROOT_DIR} ")
  message(STATUS "  CUDA_SDK_ROOT_DIR       : ${CUDA_SDK_ROOT_DIR} ")
  message(STATUS "  CUDA_VERSION            : ${CUDA_VERSION} ")
  message(STATUS "  HELPER_CUDA_INCLUDE_DIR : ${HELPER_CUDA_INCLUDE_DIR} ")
  message(STATUS "  PROJECT_SOURCE_DIR      : ${PROJECT_SOURCE_DIR} ")
  message(STATUS "  CMAKE_CURRENT_LIST_DIR  : ${CMAKE_CURRENT_LIST_DIR} ")

  message(STATUS "FindOpticksCUDA.cmake:OpticksCUDA_VERBOSE      : ${OpticksCUDA_VERBOSE} ")
  message(STATUS "FindOpticksCUDA.cmake:OpticksCUDA_FOUND        : ${OpticksCUDA_FOUND} ")
  message(STATUS "FindOpticksCUDA.cmake:OpticksHELPER_CUDA_FOUND : ${OpticksHELPER_CUDA_FOUND} ")
  message(STATUS "FindOpticksCUDA.cmake:OpticksCUDA_API_VERSION  : ${OpticksCUDA_API_VERSION} ")
  message(STATUS "FindOpticksCUDA.cmake:CUDA_LIBRARIES           : ${CUDA_LIBRARIES} ")
  message(STATUS "FindOpticksCUDA.cmake:CUDA_INCLUDE_DIRS        : ${CUDA_INCLUDE_DIRS} ")
  message(STATUS "FindOpticksCUDA.cmake:CUDA_curand_LIBRARY      : ${CUDA_curand_LIBRARY}")

   

  include(EchoTarget)
  echo_pfx_vars(CUDA "cudart_static_LIBRARY;curand_LIBRARY") 
endif()


#set(tgt Opticks::CUDA)
set(tgt Opticks::OpticksCUDA)

if(OpticksCUDA_FOUND AND NOT TARGET ${tgt})
    add_library(Opticks::cudart_static UNKNOWN IMPORTED)
    set_target_properties(Opticks::cudart_static PROPERTIES IMPORTED_LOCATION "${CUDA_cudart_static_LIBRARY}") 
    set_target_properties(Opticks::cudart_static PROPERTIES INTERFACE_IMPORTED_LOCATION "${CUDA_cudart_static_LIBRARY}") 
    # duplicate with INTERFACE_ to workaround CMake 3.13 whitelisting restriction

    add_library(Opticks::curand UNKNOWN IMPORTED)
    set_target_properties(Opticks::curand PROPERTIES IMPORTED_LOCATION "${CUDA_curand_LIBRARY}") 
    set_target_properties(Opticks::curand PROPERTIES INTERFACE_IMPORTED_LOCATION "${CUDA_curand_LIBRARY}") 
    # duplicate with INTERFACE_ to workaround CMake 3.13 whitelisting restriction

    add_library(${tgt} INTERFACE IMPORTED)
    set_target_properties(${tgt}  PROPERTIES INTERFACE_FIND_PACKAGE_NAME "OpticksCUDA MODULE REQUIRED")
    set_target_properties(${tgt}  PROPERTIES INTERFACE_PKG_CONFIG_NAME   "OpticksCUDA")

    target_link_libraries(${tgt} INTERFACE Opticks::cudart_static Opticks::curand )
    target_include_directories(${tgt} INTERFACE "${CUDA_INCLUDE_DIRS}" )

    if(OpticksHELPER_CUDA_FOUND)
        add_library(Opticks::CUDASamples INTERFACE IMPORTED)
        target_include_directories(Opticks::CUDASamples INTERFACE "${HELPER_CUDA_INCLUDE_DIR}")  
        ## for CUDA error strings from helper_cuda.h and helper_string.h 
    endif()

    set(OpticksCUDA_targets
         cudart_static
         curand
         CUDA
         CUDASamples  
    )
endif()




