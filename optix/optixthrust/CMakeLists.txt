cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OptiXThrust)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{OPTIX_SDK_DIR}/CMake"
          ) 


include(EnvBuildOptions)   

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
find_package(GLM  REQUIRED)
find_package(NPY   REQUIRED)

include(EnvCompilationFlags)  

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}/inc
   ${CUDA_INCLUDE_DIRS}
   ${OptiX_INCLUDE}  
   ${NPY_INCLUDE_DIRS}
   ${GLM_INCLUDE_DIRS} 
)

add_definitions(
   ${OptiX_DEFINITIONS} 
)

set(LIBRARIES 
      optix
      optixu
      ${CUDA_LIBRARIES}
      ${NPY_LIBRARIES}
)

# CUDA_GENERATED_OUTPUT_DIR seems to an input to OptiX cmake macros
# controlling where to put the compiled ptx, the location
# needs to match that assumed/configured within RayTraceConfig.cc
#
# set(CUDA_GENERATED_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx") 
#
# hmm 
#   for OptiX programs the cuda ptx outputs are needed by OptiX at runtime
#   but for vanilla cuda/thrust the outputs are needed for linking into libs/executables
#

message("CMAKE_CURRENT_BINARY_DIR:" ${CMAKE_CURRENT_BINARY_DIR})  
message("CUDA_GENERATED_OUTPUT_DIR:" ${CUDA_GENERATED_OUTPUT_DIR})  


optixthrust_add_library( ${name}
      optixthrust.cpp
      optixthrust_.cu     
      cu/minimal_float4.cu
      OPTIONS 
       -arch sm_30
) 

list(APPEND LIBRARIES ${name} )

optixthrust_add_executable( ${name}Test
      main.cpp
)

# list all ptx for OptiX here
set(cuptxs
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_minimal_float4.cu.ptx
)

set(hdrs 
     optixthrust.hh
)


set(TARGET    "${name}")
set(PTXDIR    "${CMAKE_INSTALL_PREFIX}/ptx")
configure_file(Config.hh.in inc/Config.hh) 

install(FILES ${cuptxs}     DESTINATION ptx)
install(FILES ${hdrs}       DESTINATION include/${name})
install(TARGETS ${name}     DESTINATION lib)
install(TARGETS ${name}Test DESTINATION bin)

