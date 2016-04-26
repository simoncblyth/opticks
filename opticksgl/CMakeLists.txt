cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

set(name OpticksGL)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${OptiX_INSTALL_DIR}/SDK/CMake") 

include(EnvBuildOptions)

########################## from OPTIX_SDK_DIR/CMake
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

include(EnvCompilationFlags)   


find_package(OpticksBoost REQUIRED)
find_package(GLM  REQUIRED)

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OGLRap REQUIRED)

find_package(NPY  REQUIRED)
find_package(OpticksCore REQUIRED)

find_package(Assimp REQUIRED)
find_package(AssimpRap REQUIRED)
find_package(GGeo REQUIRED)
find_package(CUDARap REQUIRED)
find_package(ThrustRap REQUIRED)
find_package(OptiXRap REQUIRED)
find_package(OpticksOp REQUIRED)



include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/inc   # for cmake configure_file Config.hh 

    ${OpticksBoost_INCLUDE_DIRS} 

    ${GLEW_INCLUDE_DIRS} 
    ${GLFW_INCLUDE_DIRS} 
    ${OGLRap_INCLUDE_DIRS}

    ${GLM_INCLUDE_DIRS} 
    ${NPY_INCLUDE_DIRS} 
    ${OpticksCore_INCLUDE_DIRS} 

    ${OptiX_INCLUDE}  
    ${CUDA_INCLUDE_DIRS}
    ${Assimp_INCLUDE_DIRS}  
    ${AssimpRap_INCLUDE_DIRS}  
    ${GGeo_INCLUDE_DIRS}  
    ${CUDARap_INCLUDE_DIRS}  
    ${ThrustRap_INCLUDE_DIRS}  

    ${OptiXRap_INCLUDE_DIRS}
    ${OpticksOp_INCLUDE_DIRS}
)


set(LIBRARIES
      ${OpticksBoost_LIBRARIES}

      ${GLEW_LIBRARIES}
      ${GLFW_LIBRARIES}
      ${OGLRap_LIBRARIES}

      ${GLM_LIBRARIES}
      ${NPY_LIBRARIES}
      ${OpticksCore_LIBRARIES} 

      ${CUDA_LIBRARIES}
      ${Assimp_LIBRARIES}
      ${AssimpRap_LIBRARIES}
      ${GGeo_LIBRARIES}

      ${CUDARap_LIBRARIES}
      ${ThrustRap_LIBRARIES}
      ${OptiXRap_LIBRARIES}
      ${OpticksOp_LIBRARIES}

      optix
      optixu
)

add_definitions(
    ${OpticksBoost_DEFINITIONS}

    ${GLEW_DEFINITIONS}
    ${GLFW_DEFINITIONS}
    ${OGLRap_DEFINITIONS}

    ${GLM_DEFINITIONS}
    ${NPY_DEFINITIONS}
    ${OpticksCore_DEFINITIONS} 

    ${CUDA_DEFINITIONS} 
    ${Assimp_DEFINITIONS} 
    ${AssimpRap_DEFINITIONS} 
    ${GGeo_DEFINITIONS} 

    ${OptiX_DEFINITIONS} 
    ${CUDARap_DEFINITIONS} 
    ${OptiXRap_DEFINITIONS} 
    ${OpticksOp_DEFINITIONS}
)



# NB only for classes with OpenGL dependency 
#    for others use opticksop- not opticksgl-

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
install(FILES ${hdrs}   DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
