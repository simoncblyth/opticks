#cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
cmake_minimum_required(VERSION 3.2.1 FATAL_ERROR)
set(name ThrustRap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS OFF)

# http://www.cmake.org/cmake/help/v3.0/module/FindCUDA.html

find_package(CUDA 2.3 REQUIRED)

include(EnvBuildOptions)

# following cudawrap-
# see cuda- for commentary on this morass 
#
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
#
#
# with below ... it builds but segfaults on running
# will need to update CUDA version for c++11 presumably 

message("CMAKE_CXX_COMPILER_ID:" ${CMAKE_CXX_COMPILER_ID})

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -stdlib=libstdc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -stdlib=libstdc++")
endif ()


#find_package(Boost REQUIRED)
#find_package(GLM  REQUIRED)
#find_package(NPY  REQUIRED)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
 #   ${Boost_INCLUDE_DIRS} 
 #   ${GLM_INCLUDE_DIRS} 
 #   ${NPY_INCLUDE_DIRS} 
    ${CUDA_INCLUDE_DIRS}
)

add_definitions(
  #  ${Boost_DEFINITIONS}
)

CUDA_ADD_LIBRARY( ${name}  
       ThrustEngine.cc
       histogram_sorted.cu
       make_sparse_histogram.cu
)


set(LIBRARIES
   #   ${Boost_LIBRARIES}
   #   ${NPY_LIBRARIES}
)
 
target_link_libraries( ${name} 
     ${LIBRARIES}
)

# -lc++ or -lstdc++


#set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)

SET(hdrs 
  ThrustEngine.hh
  numpy.hpp
  histogram_sorted.h
  make_sparse_histogram.h
)
#SET(cuhdrs 
#  cu/enums.h
#)
install(TARGETS ${name} DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)
#install(FILES ${cuhdrs} DESTINATION include/cu)

add_subdirectory(tests)
