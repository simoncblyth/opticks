cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name CUDARap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(PLog REQUIRED)
find_package(CUDA ${OPTICKS_CUDA_VERSION} REQUIRED QUIET)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(SysRap REQUIRED)

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PLog_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${SysRap_INCLUDE_DIRS}
)
set(DEFINITIONS
   ${PLog_DEFINITIONS}
   ${SysRap_DEFINITIONS}
)
include_directories(${INCLUDE_DIRS})
add_definitions(${DEFINITIONS})

set(LIBRARIES
   ${PLog_LIBRARIES}
   ${SysRap_LIBRARIES}
)
#message("${name}.SysRap_LIBRARIES : ${SysRap_LIBRARIES} ")

if(UNIX)
  set(LIBRARIES ${LIBRARIES} ssl )  
endif(UNIX)

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


#
# CUDA headers yield many:
#   /usr/local/cuda/include/device_functions.hpp:283:3:   warning: extension used [-Wlanguage-extension-token]
#
set(XOPT --disable-warnings) # TODO: find way to selectively disable warnings
CUDA_ADD_LIBRARY( ${name} ${SOURCES} OPTIONS -gencode=arch=compute_30,code=sm_30 ${XOPT}  )
target_link_libraries(${name} ${LIBRARIES} )

set(LIBRARIES ${LIBRARIES} ${name} )  

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


