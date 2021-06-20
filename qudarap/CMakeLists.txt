cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name QUDARap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)      

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


set(FLAGS_VERBOSE ON)

find_package(GGeo REQUIRED CONFIG)

set(OpticksCUDA_VERBOSE ON)
find_package(OpticksCUDA REQUIRED MODULE)


set(SOURCES
    QUDARAP_LOG.cc

    QU.cc
    QRng.cc  

    QGen.cc  
    QGen.cu

    QTex.cc
    QTex.cu

    QScint.cc
    QScint.cu
)

SET(HEADERS
    QUDARAP_LOG.hh
    QUDARAP_API_EXPORT.hh

    QRng.hh
    QGen.hh
    QU.hh
    QTex.hh
    QScint.hh
)

set(_LIBS
   Opticks::SysRap
   Opticks::OpticksCUDA
   Opticks::OKConf
)

set(LIBS
    Opticks::GGeo
)
if(UNIX AND NOT APPLE)
  set(LIBS ${LIBS} ssl )  
endif()


CUDA_ADD_LIBRARY( ${name} ${SOURCES} )

get_target_property(ILL ${name} INTERFACE_LINK_LIBRARIES)
message(STATUS "${name} INTERFACE_LINK_LIBRARIES:${ILL} " )

target_link_libraries( ${name} ${LIBS})
target_compile_definitions( ${name} PUBLIC OPTICKS_QUDARAP )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


