cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CUDARap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)      

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


set(FLAGS_VERBOSE ON)

find_package(SysRap REQUIRED CONFIG)
set(OpticksCUDA_VERBOSE ON)
find_package(OpticksCUDA REQUIRED MODULE)
## NB **NO BOOST** or pkgs with Boost ancestry : as nvcc barfs on it
##  this may be no longer the case 

#set(CUDARap_VERBOSE OFF)
set(CUDARap_VERBOSE ON)
if(CUDARap_VERBOSE)
   message(STATUS "${name}.CUDA_NVCC_FLAGS : ${CUDA_NVCC_FLAGS} ")
endif()



set(SOURCES
    CUDARAP_LOG.cc

    cuRANDWrapper.cc
    cuRANDWrapper_kernel.cu
    LaunchSequence.cc
    LaunchCommon.cc
    CDevice.cu
    CResource.cc
    CResource_.cu
)

SET(HEADERS

    CUDARAP_LOG.hh
    CUDARAP_API_EXPORT.hh
    CUDARAP_HEAD.hh
    CUDARAP_TAIL.hh

    LaunchCommon.hh
    LaunchSequence.hh
    cuRANDWrapper.hh
    cuRANDWrapper_kernel.hh
    CDevice.hh
    CResource.hh
    CBufSpec.hh
    CBufSlice.hh
    cfloat4x4.h
)

set(LIBRARIES
   Opticks::SysRap
   Opticks::OpticksCUDA
   Opticks::CUDASamples
   Opticks::OKConf
)
if(UNIX AND NOT APPLE)
  set(LIBRARIES ${LIBRARIES} ssl )  
endif()


CUDA_ADD_LIBRARY( ${name} ${SOURCES} )

get_target_property(ILL ${name} INTERFACE_LINK_LIBRARIES)
message(STATUS "${name} INTERFACE_LINK_LIBRARIES:${ILL} " )

#set_target_properties(${name}  PROPERTIES INTERFACE_LINK_LIBRARIES "" ) # kludge clear
target_link_libraries( ${name} ${LIBRARIES})
target_compile_definitions( ${name} PUBLIC OPTICKS_CUDARAP )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

if(CUDARap_VERBOSE)
  message(STATUS "${name}.LIBRARIES : ${LIBRARIES} ")
  echo_target_std(${name})
endif()

#install(TARGETS ${name} LIBRARY DESTINATION lib)  see notes/issues/lib64-lib-install.rst
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


#[=[
Formerly copied in helper_cuda.h from the samples distrib, now trying to avoid 
that using the Opticks::CUDASamples INTERFACE IMPORTED target.

::

  /Developer/NVIDIA/CUDA-9.1/samples/common/inc/helper_cuda.h
  /Volumes/Delta/Developer/NVIDIA/CUDA-7.0/samples/common/inc/helper_cuda.h 
  /Volumes/Delta/Developer/NVIDIA/CUDA-5.5/samples/common/inc/helper_cuda.h 


#]=]
