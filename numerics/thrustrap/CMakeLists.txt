#cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
#cmake_minimum_required(VERSION 3.2.1 FATAL_ERROR)
cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)
set(name ThrustRap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

# http://www.cmake.org/cmake/help/v3.0/module/FindCUDA.html
find_package(CUDA 2.3 REQUIRED)

include(EnvBuildOptions)

# TODO: see if can improve compilation speed by changing nvcc options to 
# target just needed device ?
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

message("CMAKE_CXX_COMPILER_ID:" ${CMAKE_CXX_COMPILER_ID})

find_package(Boost REQUIRED)
find_package(GLM  REQUIRED)
find_package(NPY  REQUIRED)
find_package(CUDAWrap  REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS} 
    ${GLM_INCLUDE_DIRS} 
    ${NPY_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
    ${CUDAWrap_INCLUDE_DIRS}
)

add_definitions(
    ${Boost_DEFINITIONS}
    ${CUDAWrap_DEFINITIONS}
)

set(LIBRARIES
      ${Boost_LIBRARIES}
      ${NPY_LIBRARIES}
      ${CUDAWrap_LIBRARIES}
)


CUDA_ADD_LIBRARY( ${name}  
       TBuf_.cu
       TBufPair_.cu
       TSparse_.cu
)
 
target_link_libraries( ${name} ${LIBRARIES})


SET(hdrs 
  TBuf.hh
  TBufPair.hh
  TSparse.hh
  strided_range.h 
  iexpand.h 
)
install(TARGETS ${name} DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

add_subdirectory(tests)
