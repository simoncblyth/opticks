cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name ThrustRap)
project(${name} VERSION 0.1.0)

include(EnvBuildOptions)
include(EnvCompilationFlags)

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)


find_package(OpticksCore REQUIRED CONFIG)
find_package(CUDARap  REQUIRED CONFIG)
 

set(SOURCES

    THRAP_LOG.cc

    TBuf_.cu
    TBufPair_.cu
    TRngBuf_.cu
    TSparse_.cu
    TUtil_.cu
)

SET(HEADERS

    THRAP_LOG.hh
    THRAP_API_EXPORT.hh
    THRAP_HEAD.hh
    THRAP_TAIL.hh

    THRAP_PUSH.hh
    THRAP_POP.hh

    TBuf.hh
    TRngBuf.hh
    TBufPair.hh
    TSparse.hh
    TUtil.hh

    strided_range.h 
    iexpand.h 
)


CUDA_ADD_LIBRARY( ${name} ${SOURCES} OPTIONS )
set_target_properties(${name}  PROPERTIES INTERFACE_LINK_LIBRARIES "" ) # kludge clear
target_link_libraries( ${name} Opticks::OpticksCore Opticks::CUDARap)
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

include(EchoTarget)
echo_target_std(${name})

install(TARGETS ${name} LIBRARY DESTINATION lib)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)



