

find_package(CUDA   REQUIRED MODULE) # eg /opt/local/share/cmake-3.11/Modules/FindCUDA.cmake

if(CUDA_LIBRARIES AND CUDA_INCLUDE_DIRS AND CUDA_curand_LIBRARY)
  set(OpticksCUDA_FOUND "YES")
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
endif()

if(OpticksCUDA_VERBOSE)

  message(STATUS "  CUDA_TOOLKIT_ROOT_DIR : ${CUDA_TOOLKIT_ROOT_DIR} ")
  message(STATUS "  CUDA_SDK_ROOT_DIR     : ${CUDA_SDK_ROOT_DIR} ")
  message(STATUS "  CUDA_VERSION          : ${CUDA_VERSION} ")

  message(STATUS "FindOpticksCUDA.cmake:OpticksCUDA_VERBOSE  : ${OpticksCUDA_VERBOSE} ")
  message(STATUS "FindOpticksCUDA.cmake:OpticksCUDA_FOUND    : ${OpticksCUDA_FOUND} ")
  message(STATUS "FindOpticksCUDA.cmake:OpticksCUDA_API_VERSION    : ${OpticksCUDA_API_VERSION} ")
  message(STATUS "FindOpticksCUDA.cmake:CUDA_LIBRARIES       : ${CUDA_LIBRARIES} ")
  message(STATUS "FindOpticksCUDA.cmake:CUDA_INCLUDE_DIRS    : ${CUDA_INCLUDE_DIRS} ")
  message(STATUS "FindOpticksCUDA.cmake:CUDA_curand_LIBRARY  : ${CUDA_curand_LIBRARY}")

  include(EchoTarget)
  echo_pfx_vars(CUDA "cudart_static_LIBRARY;curand_LIBRARY") 
endif()

if(OpticksCUDA_FOUND AND NOT TARGET Opticks::CUDA)
    add_library(Opticks::cudart_static UNKNOWN IMPORTED)
    set_target_properties(Opticks::cudart_static PROPERTIES IMPORTED_LOCATION "${CUDA_cudart_static_LIBRARY}") 

    add_library(Opticks::curand UNKNOWN IMPORTED)
    set_target_properties(Opticks::curand PROPERTIES IMPORTED_LOCATION "${CUDA_curand_LIBRARY}") 

    add_library(Opticks::CUDA INTERFACE IMPORTED)
    set_target_properties(Opticks::CUDA  PROPERTIES INTERFACE_FIND_PACKAGE_NAME "OpticksCUDA MODULE REQUIRED")

    target_link_libraries(Opticks::CUDA INTERFACE Opticks::cudart_static Opticks::curand )
    target_include_directories(Opticks::CUDA INTERFACE "${CUDA_INCLUDE_DIRS}" )

    add_library(Opticks::CUDASamples INTERFACE IMPORTED)
    target_include_directories(Opticks::CUDASamples INTERFACE "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")  
    ## for CUDA error strings from helper_cuda.h and helper_string.h 

endif()

