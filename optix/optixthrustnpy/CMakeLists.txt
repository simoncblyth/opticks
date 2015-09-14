cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OptiXThrustNPY)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{OPTIX_SDK_DIR}/CMake"
          ) 


include(EnvBuildOptions)       # OSX RPATH variable config : possibly no longer needed in current CMake version ?
set(BUILD_SHARED_LIBS ON)

################################ includes from /Developer/OptiX/SDK/CMake
include(Macros)                # string, message handling and CUDA compilation macros 
include(CompilerInfo)          # compiler detection and setting defines, nothing for gcc/clang 
include(ConfigCompilerFlags)   # flags initialized and set, nvcc absent

OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

find_package(OptiX REQUIRED)    # FindOptiX.cmake : finding and adding libraries
find_package(CUDA 2.3 REQUIRED) # 
find_package(Boost REQUIRED)
find_package(GLM  REQUIRED)
find_package(NPY   REQUIRED)
find_package(OptiXRap REQUIRED)
find_package(ThrustRap REQUIRED)

## c++11 forced by AsioZMQ   needs to come after OptiX
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++ ")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x ")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-private-field")

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CUDA_INCLUDE_DIRS}
   ${OptiX_INCLUDE}  
   ${NPY_INCLUDE_DIRS}
   ${GLM_INCLUDE_DIRS} 
   ${OptiXRap_INCLUDE_DIRS}
   ${ThrustRap_INCLUDE_DIRS}
)

add_definitions(
   ${OptiX_DEFINITIONS} 
   ${OptiXRap_DEFINITIONS}
   ${ThrustRap_DEFINITIONS}
)

set(LIBRARIES 
      optix
      optixu
      ${CUDA_LIBRARIES}
      ${NPY_LIBRARIES}
      ${OptiXRap_LIBRARIES}
      ${ThrustRap_LIBRARIES}
)

add_executable( ${name} main.cc App.cc)

target_link_libraries( ${name} ${LIBRARIES} )

install(TARGETS ${name}    DESTINATION bin)


