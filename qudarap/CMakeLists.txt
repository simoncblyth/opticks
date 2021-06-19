cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name QUDARap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)      

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


set(FLAGS_VERBOSE ON)

find_package(OpticksCore REQUIRED CONFIG)

set(OpticksCUDA_VERBOSE ON)
find_package(OpticksCUDA REQUIRED MODULE)


set(SOURCES
    QUDARAP_LOG.cc
    QRng.cc
    QRng.cu
    QU.cc
)

SET(HEADERS

    QUDARAP_LOG.hh
    QUDARAP_API_EXPORT.hh

    QRng.hh
    QU.hh

)

set(LIBRARIES
   Opticks::SysRap
   Opticks::OpticksCUDA
   Opticks::OKConf
)
if(UNIX AND NOT APPLE)
  set(LIBRARIES ${LIBRARIES} ssl )  
endif()


CUDA_ADD_LIBRARY( ${name} ${SOURCES} )

get_target_property(ILL ${name} INTERFACE_LINK_LIBRARIES)
message(STATUS "${name} INTERFACE_LINK_LIBRARIES:${ILL} " )

target_link_libraries( ${name} ${LIBRARIES})
target_compile_definitions( ${name} PUBLIC OPTICKS_QUDARAP )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )


install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)



