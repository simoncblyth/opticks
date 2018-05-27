#[=[
OptiXRap
===========

OptiXRap package needs help to find the OptiX libraries in the form of 
cmake invocation argument::

    -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir)

This seems appropriate, however currently all downstream packages
need to repeat this argument ... which seems silly in a BCM world 
if imported targets.

Finding OKConf before including OpticksBuildOptions
allows the CUDA_NVCC_FLAGS to incorporate the COMPUTE_CAPABILITY 
from the OKConf BCM generated TOPMATTER. 

#]=]
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OptiXRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

find_package(OptiX      REQUIRED MODULE)   
find_package(OpticksGeo REQUIRED CONFIG)
find_package(ThrustRap  REQUIRED CONFIG )

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
    OGeoStat.cc
    OFunc.cc
)

set(HEADERS

    OXRAP_LOG.hh
    OXRAP_API_EXPORT.hh
    OXRAP_HEAD.hh
    OXRAP_TAIL.hh

    OXRAP_PUSH.hh
    OXRAP_POP.hh

    OXPPNS.hh

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


set(CU_SOURCES

    cu/pinhole_camera.cu
    cu/constantbg.cu
    cu/material1_radiance.cu 
    cu/material1_propagate.cu 
    cu/TriangleMesh.cu 
    cu/sphere.cu 
    cu/visit_instance.cu
    cu/intersect_analytic.cu 
    cu/solve_callable.cu 
    cu/generate.cu
    cu/textureTest.cu
    cu/bufferTest.cu
    cu/dirtyBufferTest.cu
    cu/compactionTest.cu
    cu/minimalTest.cu
    cu/cbrtTest.cu
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

    cu/intersect_analytic_test.cu
    cu/Roots3And4Test.cu
)

set(CU_HEADERS

    cu/enums.h
    cu/photon.h
)

# setup nvcc compilation of OptiX RTProgram sources into PTX and CUDA sources into OBJ
CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${CU_SOURCES} )
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )  

if(OptiXRap_VERBOSE)
  message(STATUS "${name}._generated_PTX_files : ${_generated_PTX_files} ")
  message(STATUS "${name}._generated_OBJ_files : ${_generated_OBJ_files} ")
endif()


add_library( ${name} SHARED ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )
# (?) PTX are not archived in the lib, it is just expedient to list them as sources
# of the lib target so they get hooked up as dependencies, and thus are generated before 
# they need to be installed 

target_link_libraries( ${name} PUBLIC  
  Opticks::OptiX
  Opticks::OpticksGeo
  Opticks::ThrustRap
)

target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

echo_target_std(${name})

install(FILES ${_generated_PTX_files} DESTINATION installcache/PTX)
install(FILES ${CU_HEADERS}           DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cu)
install(FILES ${HEADERS}              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL )

# attempting to move TOPMATTER to OKConf
#   TOPMATTER "
#set(OptiX_INSTALL_DIR ${OptiX_INSTALL_DIR})
#message(STATUS \"\${CMAKE_CURRENT_LIST_FILE} : OptiX_INSTALL_DIR : \${OptiX_INSTALL_DIR} \")
#" )


add_subdirectory(tests)

