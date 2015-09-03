cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OptiXThrustMinimal)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
                      "$ENV{OPTIX_SDK_DIR}/CMake"
          ) 


include(EnvBuildOptions)

set(BUILD_SHARED_LIBS ON)

#####################  for OptiX
include(Macros)
include(CompilerInfo)
include(ConfigCompilerFlags)
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

find_package(OptiX REQUIRED)
find_package(CUDA 2.3 REQUIRED)

## c++11 forced by AsioZMQ   needs to come after OptiX
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")
endif ()


find_package(Boost REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CUDA_INCLUDE_DIRS}
   ${OptiX_INCLUDE}  
)

message(CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS} )

add_definitions(
   ${OptiX_DEFINITIONS} 
)

set(LIBRARIES 
      optix
      optixu
      ${CUDA_LIBRARIES}
)


function(add_optix_executable target_name)

    # split arguments into three lists 
    CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

    # KLUDGE :  hardcoding into the function
    set(obj_source_files 
        optixthrust_postprocess.cu     
    )

    # Create the rules to build the OBJ from the CUDA files.
    CUDA_WRAP_SRCS( ${target_name} OBJ obj_generated_files ${obj_source_files} ${cmake_options} OPTIONS ${options} )

    # Create the rules to build the PTX from the CUDA files.
    CUDA_WRAP_SRCS( ${target_name} PTX ptx_generated_files ${source_files} ${cmake_options} OPTIONS ${options} )

    add_executable(${target_name}
        ${source_files}
        ${obj_generated_files}
        ${ptx_generated_files}
        ${cmake_options}
    )

    target_link_libraries( ${target_name} 
        ${LIBRARIES} 
      )

endfunction()


# CUDA_GENERATED_OUTPUT_DIR seems to an input to OptiX cmake macros
# controlling where to put the compiled ptx, the location
# needs to match that assumed/configured within RayTraceConfig.cc
set(CUDA_GENERATED_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx") 
message("CUDA_GENERATED_OUTPUT_DIR:" ${CUDA_GENERATED_OUTPUT_DIR})  


#  hmm have two different flavors of .cu
#  optix programs to be made into .ptx  
#  and thrust or CUDA others to be made into .o for linkage
#
#  
#  set_source_files_properties(${SRC_LIST} PROPERTIES HEADER_FILE_ONLY 1)
#

add_optix_executable( ${name} 
      main.cpp
      optixthrust.cpp
      minimal_float4.cu
) 


install(TARGETS ${name} DESTINATION bin)



