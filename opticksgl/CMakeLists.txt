cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

set(name OpticksGL)
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
include(OptiXThrustMacros)      # for optixthrust_add_executable handling .cu partitioning


## c++11 forced by AsioZMQ : AsioZMQ not used here, but expect best to use same compiler options as far as possible
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-private-field")


find_package(Boost REQUIRED)
find_package(GLM  REQUIRED)

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OGLRap REQUIRED)

find_package(NPY  REQUIRED)
find_package(Opticks REQUIRED)

find_package(Assimp REQUIRED)
find_package(AssimpWrap REQUIRED)
find_package(GGeo REQUIRED)
find_package(CUDAWrap REQUIRED)
find_package(ThrustRap REQUIRED)
find_package(OptiXRap REQUIRED)
find_package(OpticksOp REQUIRED)

include(EnvBuildOptions)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/inc   # for cmake configure_file Config.hh 

    ${Boost_INCLUDE_DIRS} 

    ${GLEW_INCLUDE_DIRS} 
    ${GLFW_INCLUDE_DIRS} 
    ${OGLRap_INCLUDE_DIRS}

    ${GLM_INCLUDE_DIRS} 
    ${NPY_INCLUDE_DIRS} 
    ${Opticks_INCLUDE_DIRS} 

    ${OptiX_INCLUDE}  
    ${CUDA_INCLUDE_DIRS}
    ${Assimp_INCLUDE_DIRS}  
    ${AssimpWrap_INCLUDE_DIRS}  
    ${GGeo_INCLUDE_DIRS}  
    ${CUDAWrap_INCLUDE_DIRS}  
    ${ThrustRap_INCLUDE_DIRS}  

    ${OptiXRap_INCLUDE_DIRS}
    ${OpticksOp_INCLUDE_DIRS}
)


set(LIBRARIES
      ${Boost_LIBRARIES}

      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}

      ${Assimp_LIBRARIES}
      ${AssimpWrap_LIBRARIES}
      ${GGeo_LIBRARIES}
      ${CUDA_LIBRARIES}
      ${NPY_LIBRARIES}
      ${Opticks_LIBRARIES} 
      optix
      optixu
      ${CUDAWrap_LIBRARIES}
      ${ThrustRap_LIBRARIES}
      ${OptiXRap_LIBRARIES}
      ${OpticksOp_LIBRARIES}
)

add_definitions(
    ${Boost_DEFINITIONS}

    ${GLEW_DEFINITIONS}
    ${GLFW_DEFINITIONS}
    ${OGLRap_DEFINITIONS}

    ${Opticks_DEFINITIONS} 

    ${Assimp_DEFINITIONS} 
    ${OptiX_DEFINITIONS} 
    ${CUDAWrap_DEFINITIONS} 
    ${OptiXRap_DEFINITIONS} 
    ${OpticksOp_DEFINITIONS}
)



# NB only for classes with OpenGL dependency 
#    for others use opop- not opgl-

add_library( ${name}  SHARED
       OFrame.cc
       ORenderer.cc
       OpViz.cc
) 

set(hdrs 
  OFrame.hh
  ORenderer.hh
  OpViz.hh
)
 
target_link_libraries( ${name} ${LIBRARIES} )

install(TARGETS ${name} DESTINATION lib)

install(FILES ${hdrs}    DESTINATION include)

#add_subdirectory(tests)
