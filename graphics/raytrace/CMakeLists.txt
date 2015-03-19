cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
#cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

set(name RayTrace)

project(${name})



set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{OPTIX_SDK_DIR}/CMake"
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


set(BUILD_SHARED_LIBS ON)


include(Macros)

include (CompilerInfo)

include (ConfigCompilerFlags)

OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)

if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()



find_package(OptiX REQUIRED)

find_package(CUDA 2.3 REQUIRED)

find_package(AssimpWrap REQUIRED)

find_package(GGeo REQUIRED)

find_package(CUDAWrap REQUIRED)






include(FindSUtilGLUT)

include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}   # for RayTraceConfigInc.h
    ${OptiX_INCLUDE}  
    $ENV{OPTIX_SDK_DIR}/sutil
    ${CUDA_INCLUDE_DIRS}
    ${AssimpWrap_INCLUDE_DIRS}  
    ${GGeo_INCLUDE_DIRS}  
    ${CUDAWrap_INCLUDE_DIRS}  
)


add_definitions(
    ${OptiX_DEFINITIONS} 
    ${CUDAWrap_DEFINITIONS} 
)




function(add_optix_executable target_name)

    # split arguments into three lists 
    CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

    # Create the rules to build the PTX from the CUDA files.
    CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )

    add_executable(${target_name}
        ${source_files}
        ${generated_files}
        ${cmake_options}
    )


    find_library( SUTIL_LIBRARY
              NAMES sutil
              PATHS $ENV{OPTIX_SAMPLES_INSTALL_DIR}/lib )

    message("SUTIL_LIBRARY ${SUTIL_LIBRARY} ")


    target_link_libraries( ${target_name} 
                           ${SUTIL_LIBRARY}
                           optix
                           optixu
                           ${AssimpWrap_LIBRARIES} 
                           ${GGeo_LIBRARIES} 
                           ${CUDAWrap_LIBRARIES}
                           ssl
                           crypto 
                           
                         )

endfunction()


# RAYTRACE_* are written to RayTraceConfigInc.h and available as defines from the code

set(RAYTRACE_TIMEVIEW "0" )
set(RAYTRACE_CURAND   "1" )
set(RAYTRACE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Path to the raytrace source directory.")
set(RAYTRACE_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx" CACHE PATH "Path to where the raytrace code looks for nvcc generated PTX.")

# placing this inside the CMAKE_BINARY_DIR makes its prone to being wiped
set(RAYTRACE_RNG_DIR "/tmp/env/RAYTRACE_RNG_DIR" CACHE PATH "Path to where the raytrace code looks for cached rng_state.")

set(CUDA_GENERATED_OUTPUT_DIR ${RAYTRACE_PTX_DIR})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RayTraceConfigInc.h.in ${CMAKE_CURRENT_BINARY_DIR}/RayTraceConfigInc.h @ONLY)


add_optix_executable( MeshViewer
                      MeshViewer.cpp

                      RayTraceConfig.cc
                      ${CMAKE_CURRENT_BINARY_DIR}/RayTraceConfigInc.h

                      OptiXGeometry.cc
                      AssimpOptiXGeometry.cc
                      GGeoOptiXGeometry.cc

                      G4DAELoader.cc

                      cu/phong.cu
                      cu/material1_radiance.cu
                      cu/material1_touch.cu
                      cu/TriangleMesh.cu
                      
                      cu/constantbg.cu
                      cu/normal_shader.cu
                      cu/orthographic_camera.cu
                      cu/parallelogram.cu
                      cu/pinhole_camera.cu
                      cu/pinhole_camera_hybrid.cu
                      cu/touch_pinhole_camera.cu
)



 


