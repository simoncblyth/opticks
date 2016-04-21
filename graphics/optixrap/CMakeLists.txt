cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name OptiXRap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{OPTIX_SDK_DIR}/CMake"
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 


include(EnvBuildOptions)

########################## from OPTIX_SDK_DIR/CMake
include(Macros)
include(CompilerInfo)
include(ConfigCompilerFlags)

OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

find_package(OptiX REQUIRED)
find_package(CUDA 2.3 REQUIRED)
include(OptiXThrustMacros)      # for optixthrust_add_executable handling .cu partitioning

include(EnvCompilationFlags)   

find_package(Boost REQUIRED)
find_package(GLM  REQUIRED)
find_package(NPY  REQUIRED)
find_package(Opticks REQUIRED)
find_package(Assimp REQUIRED)
find_package(AssimpWrap REQUIRED)
find_package(GGeo REQUIRED)
find_package(CUDAWrap REQUIRED)
find_package(ThrustRap REQUIRED)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/inc   # for cmake configure_file Config.hh 

    ${Boost_INCLUDE_DIRS} 

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
)


set(LIBRARIES
      ${Boost_LIBRARIES}

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
)

add_definitions(
    ${Boost_DEFINITIONS}
    ${Opticks_DEFINITIONS} 
    ${Assimp_DEFINITIONS} 
    ${OptiX_DEFINITIONS} 
    ${CUDAWrap_DEFINITIONS} 
)




# OptiX prog within subdir "cu/" Thrust/CUDA not in subdir 
optixthrust_add_library( ${name} 
       OptiXUtil.cc
       OContext.cc
       OProg.cc
       OTimes.cc
       OGeo.cc

       OTracer.cc

       OColors.cc
       OBndLib.cc
       OScintillatorLib.cc
       OSourceLib.cc
       OPropertyLib.cc
       OConfig.cc
       OAccel.cc
       OPropagator.cc

       cu/pinhole_camera.cu
       cu/constantbg.cu
       cu/material1_radiance.cu 
       cu/material1_propagate.cu 
       cu/TriangleMesh.cu 
       cu/sphere.cu 
       cu/hemi-pmt.cu 
       cu/generate.cu

       OBuf_.cu     
       OBufBase_.cu     
       OBufPair_.cu     

      OPTIONS 
          -gencode=arch=compute_30,code=sm_30
) 

set(cuptxs
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_pinhole_camera.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_constantbg.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_material1_radiance.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_material1_propagate.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_TriangleMesh.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_generate.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_sphere.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_hemi-pmt.cu.ptx
)

set(cuhdrs
    cu/enums.h
    cu/photon.h
)


set(hdrs 
  OptiXUtil.hh
  OContext.hh
  OConfig.hh
  OProg.hh
  OTimes.hh
  OGeo.hh
  OTracer.hh

  OColors.hh
  OBndLib.hh
  OScintillatorLib.hh
  OSourceLib.hh
  OPropertyLib.hh
  OAccel.hh
  OPropagator.hh
  define.h

  OBuf.hh
  OBufBase.hh
  OBufPair.hh 
)
 
target_link_libraries( ${name} ${LIBRARIES} )

set(TARGET    "${name}")
set(PTXDIR    "${CMAKE_INSTALL_PREFIX}/ptx")
set(RNGDIR    "$ENV{LOCAL_BASE}/env/graphics/ggeoview/cache/rng")  # TODO: move into optixrap- OR cudawrap- fiefdom
configure_file(Config.hh.in inc/Config.hh)


install(TARGETS ${name} DESTINATION lib)
install(FILES ${cuptxs} DESTINATION ptx)
install(FILES ${hdrs}   DESTINATION include)
install(FILES ${cuhdrs} DESTINATION include/cu)

add_subdirectory(tests)
