cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

set(name OptiXRap)
project(${name})


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{OPTIX_SDK_DIR}/CMake"
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

set(BUILD_SHARED_LIBS ON)

########################## for OptiX
include(Macros)
include (CompilerInfo)
include (ConfigCompilerFlags)
OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()
find_package(OptiX REQUIRED)
find_package(CUDA 2.3 REQUIRED)


## c++11 forced by AsioZMQ : AsioZMQ not used here, but expect best to use same compiler options as far as possible
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
endif ()

# OptiXEngine rendering forces to depend on these
find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OGLRap REQUIRED)
find_package(GLM  REQUIRED)

find_package(AssimpWrap REQUIRED)
find_package(GGeo REQUIRED)
#find_package(CUDAWrap REQUIRED)


include(EnvBuildOptions)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}   # for RayTraceConfigInc.h

    ${GLEW_INCLUDE_DIRS} 
    ${GLFW_INCLUDE_DIRS} 
    ${OGLRap_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIRS} 

    ${OptiX_INCLUDE}  
    ${CUDA_INCLUDE_DIRS}
    ${AssimpWrap_INCLUDE_DIRS}  
    ${GGeo_INCLUDE_DIRS}  
)
#   ${CUDAWrap_INCLUDE_DIRS}  
#   $ENV{OPTIX_SDK_DIR}/sutil


add_definitions(
    ${GLEW_DEFINITIONS}
    ${GLFW_DEFINITIONS}
    ${OGLRap_DEFINITIONS}

    ${OptiX_DEFINITIONS} 
)
#    ${CUDAWrap_DEFINITIONS} 


add_library( ${name}  SHARED
       G4DAELoader.cc
       OptiXEngine.cc
       OptiXGeometry.cc
       GGeoOptiXGeometry.cc
       RayTraceConfig.cc
)
 
target_link_libraries( ${name} 
      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}
      ${AssimpWrap_LIBRARIES}
      ${GGeo_LIBRARIES}
      optix
      optixu
)



SET(hdrs 
  G4DAELoader.hh
  OptixEngine.hh
  OptixGeometry.hh
  GGeoOptixGeometry.hh
  RayTraceConfig.hh
)
SET(cuhdrs 
  cu/enums.h
)
install(TARGETS ${name} DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)
install(FILES ${cuhdrs} DESTINATION include/cu)



