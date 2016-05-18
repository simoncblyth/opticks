cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
set(name ThrustRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

# see cuda-/FindCuda.cmake
find_package(CUDA ${OPTICKS_CUDA_VERSION} REQUIRED QUIET)
find_package(OpticksBoost REQUIRED)
find_package(GLM  REQUIRED)
find_package(NPY  REQUIRED)
find_package(CUDARap  REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpticksBoost_INCLUDE_DIRS} 
    ${GLM_INCLUDE_DIRS} 
    ${NPY_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    ${CUDARap_INCLUDE_DIRS}
)

add_definitions(
    ${OpticksBoost_DEFINITIONS}
    ${CUDARap_DEFINITIONS}
)

set(LIBRARIES
      ${OpticksBoost_LIBRARIES}
      ${NPY_LIBRARIES}
      ${CUDARap_LIBRARIES}
)

CUDA_ADD_LIBRARY( ${name}  
       TBuf_.cu
       TBufPair_.cu
       TSparse_.cu
       TUtil_.cu
       OPTIONS -gencode=arch=compute_30,code=sm_30
)
 
target_link_libraries( ${name} ${LIBRARIES})

SET(HEADERS
  TBuf.hh
  TBufPair.hh
  TSparse.hh
  TUtil.hh
  strided_range.h 
  iexpand.h 
)
install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)
