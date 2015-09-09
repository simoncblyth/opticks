cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OptiXThrustMinimal)
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
include(OptiXThrustMacros)      # for optixthrust_add_executable handling .cu partitioning

find_package(Boost REQUIRED)

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
)

add_definitions(
   ${OptiX_DEFINITIONS} 
)

set(LIBRARIES 
      optix
      optixu
      ${CUDA_LIBRARIES}
)

# CUDA_GENERATED_OUTPUT_DIR seems to an input to OptiX cmake macros
# controlling where to put the compiled ptx, the location
# needs to match that assumed/configured within RayTraceConfig.cc
set(CUDA_GENERATED_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx") 
message("CUDA_GENERATED_OUTPUT_DIR:" ${CUDA_GENERATED_OUTPUT_DIR})  

optixthrust_add_executable( ${name} 
      main.cpp
      optixthrust.cpp
      optixthrust_.cu     
      cu/minimal_float4.cu
      OPTIONS 
       -arch sm_30
) 



install(TARGETS ${name} DESTINATION bin)



