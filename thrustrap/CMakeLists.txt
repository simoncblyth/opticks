cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
set(name ThrustRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

# see cuda-/FindCuda.cmake
find_package(CUDA ${OPTICKS_CUDA_VERSION} REQUIRED QUIET)


# Detect compute capability using cudarap- executable 
# (doesnt handle multi-GPU)
if(NOT DEFINED SM)
    find_program(_exe "cudaGetDevicePropertiesTest")
    if (_exe)
        exec_program(${_exe} ARGS q OUTPUT_VARIABLE _out RETURN_VALUE _rc)
        message("${name}.${_exe} : ${_out}")
        if (_rc EQUAL 0)
           if (_out MATCHES "^[0-9]+")
              set(SM ${_out})
           endif()
        endif()
    endif()
endif()

set(ARCH_FLAG "-arch=sm_${SM}")
list(APPEND CUDA_NVCC_FLAGS "${ARCH_FLAG};-std=c++11;-O2;-DVERBOSE")
#list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

message("${name}.CUDA_NVCC_FLAGS : ${CUDA_NVCC_FLAGS}")

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
find_package(PLog REQUIRED)
find_package(OpticksBoost REQUIRED)

find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(GLM  REQUIRED)
find_package(NPY  REQUIRED)
find_package(CUDARap  REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PLog_INCLUDE_DIRS} 
    ${OpticksBoost_INCLUDE_DIRS} 
    ${SysRap_INCLUDE_DIRS} 
    ${BoostRap_INCLUDE_DIRS} 
    ${GLM_INCLUDE_DIRS} 
    ${NPY_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    ${CUDARap_INCLUDE_DIRS}
)

add_definitions(
    ${PLog_DEFINITIONS}
    ${OpticksBoost_DEFINITIONS}
    ${SysRap_DEFINITIONS}
    ${BoostRap_DEFINITIONS}
    ${CUDARap_DEFINITIONS}
)

set(LIBRARIES
    ${PLog_LIBRARIES}
    ${OpticksBoost_LIBRARIES}
    ${SysRap_LIBRARIES}
    ${BoostRap_LIBRARIES}
    ${NPY_LIBRARIES}
    ${CUDARap_LIBRARIES}
)

set(SOURCES

    THRAP_LOG.cc

    TBuf_.cu
    TBufPair_.cu
    TSparse_.cu
    TUtil_.cu
)

SET(HEADERS

    THRAP_LOG.hh
    THRAP_API_EXPORT.hh
    THRAP_HEAD.hh
    THRAP_TAIL.hh

    THRAP_PUSH.hh
    THRAP_POP.hh

    TBuf.hh
    TBufPair.hh
    TSparse.hh
    TUtil.hh

    strided_range.h 
    iexpand.h 
)

set(XOPT --disable-warnings)
CUDA_ADD_LIBRARY( ${name} ${SOURCES} OPTIONS -gencode=arch=compute_30,code=sm_30 ${XOPT})
target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


