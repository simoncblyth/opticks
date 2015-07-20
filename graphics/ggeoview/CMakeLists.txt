cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
#cmake_policy(SET CMP0054 OLD)
# unfortunately this doesnt suppress the warnings, despite being advertised to do so
# http://www.cmake.org/Wiki/CMake/Policies
set(name GGeoView)
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

OPTION(NPYSERVER "using the numpyserver." OFF)

if (NPYSERVER)
    add_definitions(-DNPYSERVER)
endif()

find_package(OptiX REQUIRED)
find_package(CUDA 2.3 REQUIRED)


## c++11 forced by AsioZMQ   needs to come after OptiX
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif ()


########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OGLRap REQUIRED)
find_package(Boost REQUIRED)

if (NPYSERVER)
find_package(ZMQ REQUIRED)  
find_package(AsioZMQ REQUIRED)
find_package(NumpyServer REQUIRED)
endif()

find_package(NPY   REQUIRED)
find_package(GLM  REQUIRED)
find_package(Cfg REQUIRED)

find_package(Assimp REQUIRED)
find_package(AssimpWrap REQUIRED)
find_package(OptiXRap REQUIRED)
find_package(GGeo REQUIRED)
find_package(CUDAWrap REQUIRED)

find_package(ImGui REQUIRED)
find_package(Bregex REQUIRED)
find_package(ThrustRap REQUIRED)



include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${OGLRap_INCLUDE_DIRS} 
   ${Boost_INCLUDE_DIRS} 
   ${ZMQ_INCLUDE_DIRS} 
   ${AsioZMQ_INCLUDE_DIRS} 
   ${NumpyServer_INCLUDE_DIRS} 
   ${NPY_INCLUDE_DIRS} 
   ${Cfg_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 

   ${Assimp_INCLUDE_DIRS}  
   ${AssimpWrap_INCLUDE_DIRS}  
   ${CUDA_INCLUDE_DIRS}
   ${OptiX_INCLUDE}  
   ${OptiXRap_INCLUDE_DIRS}  
   ${GGeo_INCLUDE_DIRS}  
   ${CUDAWrap_INCLUDE_DIRS}  
   ${ImGui_INCLUDE_DIRS}
   ${Bregex_INCLUDE_DIRS}
   ${ThrustRap_INCLUDE_DIRS}
)


message(CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS} )

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${OGLRap_DEFINITIONS}
   ${Assimp_DEFINITIONS}  
   ${AssimpWrap_DEFINITIONS}  
   ${OptiX_DEFINITIONS} 
   ${OptiXRap_DEFINITIONS} 
   ${Boost_DEFINITIONS} 
   ${CUDAWrap_DEFINITIONS}
   ${ImGui_DEFINITIONS}
   ${Bregex_DEFINITIONS}
   ${ThrustRap_DEFINITIONS}
)


set(LIBRARIES 

      optix
      optixu

      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}
      ${Boost_LIBRARIES}
      ${ZMQ_LIBRARIES}
      ${Cfg_LIBRARIES}
      ${NumpyServer_LIBRARIES}
      ${NPY_LIBRARIES}

      ${Assimp_LIBRARIES} 
      ${AssimpWrap_LIBRARIES} 
      ${GGeo_LIBRARIES} 
      ${OptiXRap_LIBRARIES}
      ${GGeo_LIBRARIES}
      ${CUDA_LIBRARIES}
      ${CUDAWrap_LIBRARIES}
      ${ImGui_LIBRARIES}
      ${Bregex_LIBRARIES}
      ${ThrustRap_LIBRARIES}

      ssl
      crypto 

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

    target_link_libraries( ${target_name} 
        ${LIBRARIES} 
      )

endfunction()


# CUDA_GENERATED_OUTPUT_DIR seems to an input to OptiX cmake macros
# controlling where to put the compiled ptx, the location
# needs to match that assumed/configured within RayTraceConfig.cc
set(CUDA_GENERATED_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib/ptx") 
message("CUDA_GENERATED_OUTPUT_DIR:" ${CUDA_GENERATED_OUTPUT_DIR})  


add_optix_executable( ${name} 
      main.cc 
      CUDAInterop.cc
      index.cu

      cu/pinhole_camera.cu
      cu/constantbg.cu
      cu/material1_radiance.cu 
      cu/material1_propagate.cu 
      cu/TriangleMesh.cu 

      cu/generate.cu
      
) 


set(cuhdrs
    cu/photon.h
)


install(TARGETS ${name} DESTINATION bin)
install(FILES ${cuhdrs} DESTINATION include/cu)



add_subdirectory(tests)
