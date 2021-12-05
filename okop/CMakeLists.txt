cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OKOP) 
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions NO_POLICY_SCOPE)   

#if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
#    cmake_policy(SET CMP0077 OLD)
#endif()


set(FLAGS_VERBOSE ON)
find_package(OptiXRap REQUIRED CONFIG)

set(SOURCES
     
    OKOP_LOG.cc

    OpZeroer.cc
    OpSeeder.cc
    OpIndexer.cc
    OpIndexer_.cu
    OpIndexerApp.cc
    OpEngine.cc
    OpMgr.cc
    OpEvt.cc
    OpPropagator.cc
    OpTracer.cc
)

set(HEADERS

    OKOP_LOG.hh
    OKOP_API_EXPORT.hh
    OKOP_HEAD.hh
    OKOP_TAIL.hh

    OpZeroer.hh
    OpSeeder.hh
    OpIndexer.hh
    OpIndexerApp.hh
    OpEngine.hh
    OpPropagator.hh
    OpTracer.hh
    OpMgr.hh
)

# setup nvcc compilation of CUDA sources into OBJ
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )  
add_library( ${name} SHARED ${_generated_OBJ_files} ${SOURCES} )

target_link_libraries( ${name} PUBLIC Opticks::OptiXRap)
target_compile_definitions( ${name} PUBLIC OPTICKS_OKOP )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

if(OKOP_VERBOSE)
  echo_target_std(${name})
  message(STATUS "${name}._generated_OBJ_files : ${_generated_OBJ_files} ")
endif()

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)

