cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
#cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

set(name RayTrace)

project(${name})



set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{OPTIX_SDK_DIR}/CMake"
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


set(BUILD_SHARED_LIBS ON)


# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

# Turn off the warning that NVCC issues when generating PTX from our CUDA samples.  This
# is a custom extension to the FindCUDA code distributed by CMake.
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)


# For Xcode 5, gcc is actually clang, so we have to tell CUDA to treat the compiler as
# clang, so that it doesn't mistake it for something else.
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()




find_package(OptiX REQUIRED)
find_package(CUDA 2.3 REQUIRED)
find_package(Assimp REQUIRED)


include(FindSUtilGLUT)

include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OptiX_INCLUDE}  
    $ENV{OPTIX_SDK_DIR}/sutil
    ${CUDA_INCLUDE_DIRS}
    ${Assimp_INCLUDE_DIRS}  
)

add_definitions(
    ${OptiX_DEFINITIONS} 
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
                           ${Assimp_LIBRARIES} 
                           ssl
                           crypto 
                         )

endfunction()


set(RAYTRACE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Path to the raytrace source directory.")
set(RAYTRACE_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx" CACHE PATH "Path to where the raytrace code looks for nvcc generated PTX.")
set(CUDA_GENERATED_OUTPUT_DIR ${RAYTRACE_PTX_DIR})
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/RayTraceConfigInc.h.in ${CMAKE_CURRENT_BINARY_DIR}/RayTraceConfigInc.h @ONLY)


#add_optix_executable(${name} 
#                     ${name}.cc 
#                     RayTraceConfig.cc
#                     ${CMAKE_CURRENT_BINARY_DIR}/RayTraceConfigInc.h
#                     AssimpGeometry.cc
#                     AssimpCommon.cc
#                     AssimpNode.cc
#                     AssimpTree.cc
#                     AssimpRegistry.cc
#                     OptiXAssimpGeometry.cc
#                     OptiXScene.cc
#                     tutorial0.cu
#                     TriangleMesh.cu
#                     material1.cu
#)

add_optix_executable( MeshViewer
                      MeshViewer.cpp
                      RayTraceConfig.cc
                      ${CMAKE_CURRENT_BINARY_DIR}/RayTraceConfigInc.h
                      AssimpGeometry.cc
                      AssimpCommon.cc
                      AssimpNode.cc
                      AssimpTree.cc
                      AssimpRegistry.cc
                      OptiXAssimpGeometry.cc
                      G4DAELoader.cc

                      materials/material1.cu
                      TriangleMesh.cu
                      
                      sample6/accum_camera.cu
                      sample6/ambocc.cu
                      sample6/constantbg.cu
                      sample6/normal_shader.cu
                      sample6/one_bounce_diffuse.cu
                      sample6/orthographic_camera.cu
                      sample6/parallelogram.cu
                      sample6/pinhole_camera.cu
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})  # for RayTraceConfigInc.h



# non-optix geometry checking 
add_executable( AssimpGeometryTest 
                AssimpGeometryTest.cc
                AssimpGeometry.cc
                AssimpCommon.cc
                AssimpNode.cc
                AssimpTree.cc
                AssimpRegistry.cc
)

target_link_libraries( AssimpGeometryTest 
                       ${Assimp_LIBRARIES} 
                     )
                        



 


