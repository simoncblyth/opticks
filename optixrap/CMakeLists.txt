cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name OptiXRap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${OptiX_INSTALL_DIR}/SDK/CMake") 

include(EnvBuildOptions)

find_package(OptiX ${OPTICKS_OPTIX_VERSION} REQUIRED)


########################## from OPTIX_SDK_DIR/CMake
include(Macros)
include(CompilerInfo)
include(ConfigCompilerFlags)

OPTION(CUDA_REMOVE_GLOBAL_MEMORY_SPACE_WARNING "Suppress the \"Advisory: Cannot tell what pointer points to, assuming global memory space\" warning nvcc makes." ON)
if(USING_CLANG_C)
  set(CUDA_HOST_COMPILER "clang" CACHE FILEPATH "Host side compiler used by NVCC")
endif()

########################

find_package(CUDA ${OPTICKS_CUDA_VERSION} REQUIRED QUIET)
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O2;-DVERBOSE")
#list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
#list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
include(OptiXThrustMacros)      # for optixthrust_add_executable handling .cu partitioning

include(EnvCompilationFlags)   

# In SUSE Linux + gcc 4.8, if the pedantic exists, the compilation will be failed.
string (REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

find_package(PLog REQUIRED)
find_package(OpticksBoost REQUIRED)
find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(GLM  REQUIRED)
find_package(NPY  REQUIRED)
find_package(OpticksCore REQUIRED)

find_package(Assimp REQUIRED)
find_package(AssimpRap REQUIRED)
find_package(GGeo REQUIRED)
find_package(CUDARap REQUIRED)
find_package(ThrustRap REQUIRED)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/inc   # for cmake configure_file Config.hh 

    ${PLog_INCLUDE_DIRS} 
    ${OpticksBoost_INCLUDE_DIRS} 
    ${SysRap_INCLUDE_DIRS} 
    ${BoostRap_INCLUDE_DIRS} 

    ${GLM_INCLUDE_DIRS} 
    ${NPY_INCLUDE_DIRS} 
    ${OpticksCore_INCLUDE_DIRS} 

    ${OptiX_INCLUDE_DIRS}  

    ${CUDA_INCLUDE_DIRS}
    ${Assimp_INCLUDE_DIRS}  
    ${AssimpRap_INCLUDE_DIRS}  
    ${GGeo_INCLUDE_DIRS}  
    ${CUDARap_INCLUDE_DIRS}  
    ${ThrustRap_INCLUDE_DIRS}  
)


set(LIBRARIES
      ${PLog_LIBRARIES}
      ${OpticksBoost_LIBRARIES}
      ${SysRap_LIBRARIES} 
      ${BoostRap_LIBRARIES} 

      ${Assimp_LIBRARIES}
      ${AssimpRap_LIBRARIES}
      ${GGeo_LIBRARIES}
      ${CUDA_LIBRARIES}
      ${NPY_LIBRARIES}
      ${OpticksCore_LIBRARIES} 

      ${OptiX_LIBRARIES}

      ${CUDARap_LIBRARIES}
      ${ThrustRap_LIBRARIES}
)

add_definitions(
    ${PLog_DEFINITIONS}
    ${OpticksBoost_DEFINITIONS}
    ${SysRap_DEFINITIONS} 
    ${BoostRap_DEFINITIONS} 
    ${OpticksCore_DEFINITIONS} 
    ${Assimp_DEFINITIONS} 
    ${AssimpRap_DEFINITIONS} 
    ${OptiX_DEFINITIONS} 
    ${CUDARap_DEFINITIONS} 
)



## OGeo needs _API template instanciation

set(SOURCES
     
    OXRAP_LOG.cc

    OptiXUtil.cc
    OAccel.cc
    OProg.cc
    OTimes.cc
    OConfig.cc
    OContext.cc
    OColors.cc
    OPropertyLib.cc
    OBndLib.cc
    OScintillatorLib.cc
    OSourceLib.cc
    OTracer.cc
    OEngineImp.cc

    OBuf_.cu     
    OBufBase_.cu     
    OBufPair_.cu     
    OPropagator.cc
    OTextureTest.cc

    OGeo.cc

    cu/pinhole_camera.cu
    cu/constantbg.cu
    cu/material1_radiance.cu 
    cu/material1_propagate.cu 
    cu/TriangleMesh.cu 
    cu/sphere.cu 
    cu/hemi-pmt.cu 
    cu/generate.cu
    cu/textureTest.cu
)


set(OX_PTXS
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_pinhole_camera.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_constantbg.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_material1_radiance.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_material1_propagate.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_TriangleMesh.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_sphere.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_hemi-pmt.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_generate.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_textureTest.cu.ptx
)


set(HEADERS

    OXRAP_LOG.hh
    OXRAP_API_EXPORT.hh
    OXRAP_HEAD.hh
    OXRAP_TAIL.hh

    OXRAP_PUSH.hh
    OXRAP_POP.hh


    OptiXUtil.hh
    OContext.hh
    OConfig.hh
    OProg.hh
    OTimes.hh
    OGeo.hh
    OTracer.hh
    OEngineImp.hh

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


set(OX_HEADERS

    cu/enums.h
    cu/photon.h
)


# OptiX prog within "cu/"  Thrust/CUDA not in "cu/"

set(XOPT --disable-warnings)

if(APPLE)
   set(NOPT -gencode=arch=compute_30,code=sm_30)
else(APPLE)
   set(NOPT)
endif(APPLE)

optixthrust_add_library( ${name} ${SOURCES} OPTIONS ${NOPT} ${XOPT}) 
target_link_libraries( ${name} ${LIBRARIES} )

set(TARGET    "${name}")
set(PTXDIR    "${CMAKE_INSTALL_PREFIX}/ptx")
set(RNGDIR    "$ENV{LOCAL_BASE}/opticks/cache/rng")  # TODO: move into optixrap- OR cudawrap- fiefdom
configure_file(Config.hh.in inc/Config.hh)


install(TARGETS ${name}     DESTINATION lib)
install(FILES ${OX_PTXS}    DESTINATION ptx)
install(FILES ${OX_HEADERS} DESTINATION include/${name}/cu)
install(FILES ${HEADERS}    DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


