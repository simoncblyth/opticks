cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CUDARap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)      

set(OKConf_VERBOSE ON)
message(STATUS "${name}.finding OKConf")
find_package(OKConf REQUIRED CONFIG)   
message(STATUS "${name}.COMPUTE_CAPABILITY ${COMPUTE_CAPABILITY} ")


set(FLAGS_VERBOSE ON)
include(OpticksCompilationFlags)     ## repeating this here as nvcc flags require COMPUTE_CAPABILIY from OKConf     

find_package(SysRap REQUIRED CONFIG)
find_package(OpticksCUDA REQUIRED MODULE)

## NB **NO BOOST** or pkgs with Boost in its ancestry : as nvcc barfs on it

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
    CResource.hh
    CBufSpec.hh
    CBufSlice.hh
)

set(LIBRARIES
   Opticks::SysRap
   Opticks::CUDA
   Opticks::CUDASamples
   Opticks::OKConf
)
if(UNIX AND NOT APPLE)
  set(LIBRARIES ${LIBRARIES} ssl )  
endif()


CUDA_ADD_LIBRARY( ${name} ${SOURCES} )
#set_target_properties(${name}  PROPERTIES INTERFACE_LINK_LIBRARIES "" ) # kludge clear
target_link_libraries( ${name} ${LIBRARIES})
target_compile_definitions( ${name} PUBLIC OPTICKS_CUDARAP )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

if(CUDARap_VERBOSE)
  message(STATUS "${name}.LIBRARIES : ${LIBRARIES} ")
  echo_target_std(${name})
endif()

install(TARGETS ${name} LIBRARY DESTINATION lib)
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
