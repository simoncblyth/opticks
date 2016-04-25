cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name CUDARap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(CUDA 2.3 REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

CUDA_ADD_LIBRARY( ${name} 
                  md5digest.cc
                  cuRANDWrapper.cc
                  cuRANDWrapper_kernel.cu
                  LaunchSequence.cc
                  LaunchCommon.cc
                  CResource.cc
                  CResource_.cu
                 OPTIONS
                     -gencode=arch=compute_30,code=sm_30                  
)

set (LIBRARIES
    ${name}
    ssl
 #   crypt
)


# TODO: adopt standard tests approach 
add_executable( LaunchSequenceTest LaunchSequenceTest.cc )
target_link_libraries( LaunchSequenceTest ${LIBRARIES} )

add_executable( cuRANDWrapperTest cuRANDWrapperTest.cc )
target_link_libraries( cuRANDWrapperTest ${LIBRARIES})


install(TARGETS ${name} LIBRARY DESTINATION lib)
install(TARGETS LaunchSequenceTest DESTINATION bin)
install(TARGETS cuRANDWrapperTest  DESTINATION bin)

SET(hdrs 
LaunchCommon.hh
LaunchSequence.hh
cuRANDWrapper.hh
cuRANDWrapper_kernel.hh
CResource.hh
CBufSpec.hh
CBufSlice.hh
)

install(FILES ${hdrs} DESTINATION include/${name})


