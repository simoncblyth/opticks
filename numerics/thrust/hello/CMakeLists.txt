cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name ThrustHello)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)
set(BUILD_SHARED_LIBS ON)

## c++11 forced by AsioZMQ
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -stdlib=libstdc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
endif ()

set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA)
find_package(CUDA 2.3 REQUIRED)

# remove -gencode
#list( APPEND CUDA_NVCC_FLAGS -link arch=compute_30,code=sm_30 -DTHRUST_DEVICE_SYSTEM=${THRUST_DEVICE_SYSTEM} )
#list(APPEND CUDA_NVCC_FLAGS -ccbin /usr/bin/clang -arch=sm_20  )
#set(CUDA_NVCC_FLAGS -ccbin /usr/bin/clang -gencode arch=compute_30,code=sm_30 )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

message("CUDA_NVCC_FLAGS" ${CUDA_NVCC_FLAGS})


include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CUDA_INCLUDE_DIRS}
)

add_definitions(
)

set(LIBRARIES
)


function(add_thrust_executable target_name)

    # split arguments into three lists 
    CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

    # Create the rules to build the PTX from the CUDA files.
    #CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )

    message("source_files:" ${source_files})

    add_executable(${target_name}
        ${source_files}
     #   ${generated_files}
        ${cmake_options}
    )

    target_link_libraries( ${target_name} 
        ${LIBRARIES} 
        ${CUDA_LIBRARIES}
      )

endfunction()


# CUDA_GENERATED_OUTPUT_DIR seems to an input to OptiX cmake macros
# controlling where to put the compiled ptx, the location
# needs to match that assumed/configured within RayTraceConfig.cc
#set(CUDA_GENERATED_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx") 
message("CUDA_GENERATED_OUTPUT_DIR:" ${CUDA_GENERATED_OUTPUT_DIR})  






set(srcs
     helloThrust.cc
     helloa.cc
     hello.cu
)

#add_thrust_executable( ${name} ${srcs} )

cuda_add_executable( ${name} ${srcs} )

target_link_libraries( ${name} ${LIBRARIES} )

#set_target_properties(${name} PROPERTIES LINKER_LANGUAGE CXX)

install(TARGETS ${name} DESTINATION bin)





add_subdirectory(tests)
