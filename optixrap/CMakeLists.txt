cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)
set(name OptiXRap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${OptiX_INSTALL_DIR}/SDK/CMake") 

include(EnvBuildOptions)

find_package(OptiX ${OPTICKS_OPTIX_VERSION} REQUIRED)   



###### Find #define OPTIX_VERSION by parsing optix.h set variable at parent scope 
#
# OptiX not playing ball with CMake version conventions, even EXACT has no teeth, 
# so parse the optix.h header to get the #define OPTIX_VERSION into CMake variable
# This means can know the version at configure time
# the value is written to inc/OpticksCMakeConfig.hh by top level CMakeLists.txt

set(OXRAP_OPTIX_VERSION 0 PARENT_SCOPE)
if(OptiX_FOUND)
   message(STATUS "${name}.OPTICKS_OPTIX_VERSION : ${OPTICKS_OPTIX_VERSION}  FOUND ")
   message(STATUS "${name}.OptiX_INCLUDE_DIRS    : ${OptiX_INCLUDE_DIRS} ")
   message(STATUS "${name}.OptiX_LIBRARIES       : ${OptiX_LIBRARIES} ")
   file(READ "${OptiX_INCLUDE_DIRS}/optix.h" _contents)
   #message(STATUS "${name}.contents : ${_contents} ")
   string(REGEX REPLACE "\n" ";" _contents "${_contents}")
   foreach(_line ${_contents})
        if (_line MATCHES "#define OPTIX_VERSION ([0-9]+) ")
            set(OXRAP_OPTIX_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
            message(STATUS "${name}._line ${_line} ===> ${CMAKE_MATCH_1} ") 
        endif()
   endforeach()
else(OptiX_FOUND)
   message(STATUS "${name}.OPTICKS_OPTIX_VERSION : ${OPTICKS_OPTIX_VERSION}  NOT-FOUND ")
endif(OptiX_FOUND)


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
find_package(OpticksGeometry REQUIRED)

find_package(CUDA ${OPTICKS_CUDA_VERSION} REQUIRED QUIET)
find_package(CUDARap REQUIRED)
find_package(ThrustRap REQUIRED)

#    ${CMAKE_CURRENT_BINARY_DIR}/inc   # for cmake configure_file Config.hh 

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}

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
    ${OpticksGeometry_INCLUDE_DIRS}  
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
      ${OpticksGeometry_LIBRARIES}  
      ${CUDA_LIBRARIES}
      ${NPY_LIBRARIES}
      ${OpticksCore_LIBRARIES} 

      ${OptiX_LIBRARIES}

      ${CUDA_LIBRARIES}
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
    OptiXTest.cc
    OAccel.cc
    OProg.cc
    OConfig.cc
    OContext.cc
    OColors.cc
    OPropertyLib.cc
    OBndLib.cc
    OScintillatorLib.cc
    OSourceLib.cc
    OTracer.cc
    OScene.cc

    OBuf_.cu     
    OBufBase_.cu     
    OBufPair_.cu     
    OPropagator.cc
    ORng.cc
    OEvent.cc
    OLaunchTest.cc

    OGeo.cc

    cu/pinhole_camera.cu
    cu/constantbg.cu
    cu/material1_radiance.cu 
    cu/material1_propagate.cu 
    cu/TriangleMesh.cu 
    cu/sphere.cu 
    cu/intersect_analytic.cu 
    cu/generate.cu
    cu/textureTest.cu
    cu/bufferTest.cu
    cu/dirtyBufferTest.cu
    cu/compactionTest.cu
    cu/minimalTest.cu
    cu/LTminimalTest.cu
    cu/axisTest.cu
    cu/seedTest.cu
    cu/boundaryTest.cu
    cu/boundaryLookupTest.cu
    cu/tex0Test.cu
    cu/texTest.cu
    cu/OEventTest.cu
    cu/OInterpolationTest.cu
    cu/ORayleighTest.cu
)


set(OX_PTXS
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_pinhole_camera.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_constantbg.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_material1_radiance.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_material1_propagate.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_TriangleMesh.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_sphere.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_intersect_analytic.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_generate.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_textureTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_minimalTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_LTminimalTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_axisTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_seedTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_boundaryTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_boundaryLookupTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_tex0Test.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_texTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_bufferTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_dirtyBufferTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_compactionTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_OEventTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_OInterpolationTest.cu.ptx
    ${CMAKE_CURRENT_BINARY_DIR}/${name}_generated_ORayleighTest.cu.ptx
)


set(HEADERS

    OXRAP_LOG.hh
    OXRAP_API_EXPORT.hh
    OXRAP_HEAD.hh
    OXRAP_TAIL.hh

    OXRAP_PUSH.hh
    OXRAP_POP.hh


    OptiXUtil.hh
    OptiXTest.hh
    OContext.hh
    OConfig.hh
    OProg.hh
    OGeo.hh
    OTracer.hh
    OScene.hh

    OColors.hh
    OBndLib.hh
    OScintillatorLib.hh
    OSourceLib.hh
    OPropertyLib.hh
    OAccel.hh
    OPropagator.hh
    ORng.hh
    OEvent.hh
    numquad.h

    OBuf.hh
    OBufBase.hh
    OBufPair.hh 
)


set(OX_HEADERS

    cu/enums.h
    cu/photon.h
)


# OptiX prog within "cu/"  Thrust/CUDA not in "cu/"

optixthrust_add_library( ${name} ${SOURCES} OPTIONS ) 
target_link_libraries( ${name} ${LIBRARIES} )

set(TARGET    "${name}")

#set(PTXDIR    "${CMAKE_INSTALL_PREFIX}/installcache/ptx")
#set(RNGDIR    "${CMAKE_INSTALL_PREFIX}/installcache/RNG")  ## duplicates cudarap- envvar and BOpticksResource value
#configure_file(Config.hh.in inc/Config.hh)

## need to match paths in brap-/BOpticksResource

install(TARGETS ${name}     DESTINATION lib)
install(FILES ${OX_PTXS}    DESTINATION installcache/PTX)
install(FILES ${OX_HEADERS} DESTINATION include/${name}/cu)
install(FILES ${HEADERS}    DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


