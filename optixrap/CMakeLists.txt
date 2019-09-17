cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OptiXRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


find_package(OKConf     REQUIRED CONFIG)  

set(OptiX_VERBOSE ON)
find_package(OptiX      REQUIRED MODULE)   

find_package(OpticksGeo REQUIRED CONFIG)
find_package(ThrustRap  REQUIRED CONFIG )


set(SOURCES
     
    OXRAP_LOG.cc
    OError.cc

    OptiXUtil.cc
    OptiXTest.cc
    OAccel.cc
    OProg.cc
    OConfig.cc
    OFormat.cc
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

    OGeometry.cc
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
    OError.hh

    OptiXUtil.hh
    OptiXTest.hh
    OContext.hh
    OConfig.hh
    OFormat.hh
    OProg.hh
    OGeometry.hh
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
    cu/GeometryTriangles.cu
    cu/constantbg.cu
    cu/material1_radiance.cu 
    cu/material1_propagate.cu 
    cu/TriangleMesh.cu 
    cu/sphere.cu 
    cu/visit_instance.cu
    cu/intersect_analytic.cu 
    cu/solve_callable.cu 
    cu/generate.cu

    cu/dirtyBufferTest.cu
    cu/compactionTest.cu
    cu/LTminimalTest.cu
    cu/axisTest.cu
    cu/seedTest.cu

    cu/cbrtTest.cu
    cu/Roots3And4Test.cu
)



set( FORMER_CU_SOURCES_NOW_MIGRATED_TO_TESTS

    cu/bufferTest.cu
    cu/textureTest.cu
    cu/boundaryTest.cu
    cu/boundaryLookupTest.cu
    cu/texTest.cu
    cu/tex0Test.cu
    cu/minimalTest.cu
    cu/ORayleighTest.cu
    cu/OEventTest.cu
    cu/OInterpolationTest.cu

    cu/intersect_analytic_dummy_test.cu
    cu/intersect_analytic_torus_test.cu
    cu/intersect_analytic_sphere_test.cu
    cu/intersect_analytic_cone_test.cu
    cu/intersect_analytic_convexpolyhedron_test.cu
)


set(CU_HEADERS

    cu/enums.h
    cu/photon.h
)

set(SCRIPTS
    tests/interpolationTest_interpol.py
    tests/interpolationTest_identity.py 
)



# setup nvcc compilation of OptiX RTProgram sources into PTX and CUDA sources into OBJ
CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${CU_SOURCES} )
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )  

add_library( ${name} SHARED ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )
#[=[
The PTX are not archived in the lib, it is just expedient to list them as sources
of the lib target so they get hooked up as dependencies, and thus are generated before 
they need to be installed 
#]=]


target_link_libraries( ${name} PUBLIC  
  Opticks::OKConf
  Opticks::OptiX
  Opticks::OpticksGeo
  Opticks::ThrustRap
)

#[=[
Target order in target_link_libraries follows thru to the 
generated config which is used by consumers of the exported target.:

  $(opticks-prefix)/lib/cmake/optixrap/optixrap-config.cmake

Thus as finding OKConf has the side effect of defining the 
OptiX_INSTALL_DIR that is needed to find OptiX the OKConf must 
come before the OptiX.

#]=]

target_compile_definitions( ${name} PUBLIC OPTICKS_OXRAP )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

if(OptiXRap_VERBOSE)
  message(STATUS "${name}._generated_PTX_files : ${_generated_PTX_files} ")
  message(STATUS "${name}._generated_OBJ_files : ${_generated_OBJ_files} ")
  echo_target_std(${name})
endif()

install(FILES ${_generated_PTX_files} DESTINATION installcache/PTX)
install(FILES ${CU_HEADERS}           DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cu)
install(FILES ${HEADERS}              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(PROGRAMS ${SCRIPTS}           DESTINATION bin)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL )

add_subdirectory(tests)

