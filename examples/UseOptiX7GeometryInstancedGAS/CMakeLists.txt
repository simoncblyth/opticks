cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOptiX7GeometryInstancedGAS)
project(${name})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(CUDA REQUIRED)   
find_package(OptiX REQUIRED)   

#[=[
Notice no OptiX libs anymore in 7, 
just headers giving access to functionality from the driver ?


    ${OptiX_INCLUDE}/../SDK

#]=]


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OptiX_INCLUDE}
    ${CMAKE_INSTALL_PREFIX}/externals/glm/glm
    ${CMAKE_CURRENT_SOURCE_DIR} 
)


set(COMPUTE_CAPABILITY 70)

set(CUDA_NVCC_FLAGS)
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
list(APPEND CUDA_NVCC_FLAGS "-O2")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")



set(SOURCES
    ${name}.cc
    Engine.cc 
    GAS.cc 
    GAS_Builder.cc 
    PIP.cc 
    IAS.cc 
    IAS_Builder.cc 
    Geo.cc
)

set(OPTIX_SOURCES
    ${name}.cu
)

CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${OPTIX_SOURCES} )
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )

message(STATUS "_generated_OBJ_files ${_generated_OBJ_files}") 
message(STATUS "_generated_PTX_files ${_generated_PTX_files}") 


add_executable( ${name} ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )


target_link_libraries(${name} 
     PUBLIC 
         ${CUDA_LIBRARIES}
)
install(TARGETS ${name}               DESTINATION bin)
install(FILES ${_generated_PTX_files} DESTINATION ptx)


