cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

set(name CUDAWrap)

project(${name})


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


set(BUILD_SHARED_LIBS ON)

find_package(CUDA 2.3 REQUIRED)

include(EnvBuildOptions)

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
)

add_executable( LaunchSequenceTest LaunchSequenceTest.cc )
target_link_libraries( LaunchSequenceTest ${name})

add_executable( cuRANDWrapperTest cuRANDWrapperTest.cc )
target_link_libraries( cuRANDWrapperTest ${name})


install(TARGETS ${name} LIBRARY DESTINATION lib)
install(TARGETS LaunchSequenceTest DESTINATION bin)
install(TARGETS cuRANDWrapperTest  DESTINATION bin)

SET(hdrs 
LaunchCommon.hh
LaunchSequence.hh
cuRANDWrapper.hh
cuRANDWrapper_kernel.hh
)

install(FILES ${hdrs} DESTINATION include)


