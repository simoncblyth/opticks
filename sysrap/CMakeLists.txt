cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name SysRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

#[=[
SysRap
========

The default RPATH setup from OpticksBuildOptions of 
"set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)"
appears to only work if your executables link to a library 
which is in the same directory as the library you are creating.  
Fortunately thats easy to do by linking with Opticks::OKConf.

#]=]

find_package(OKConf REQUIRED CONFIG)
find_package(PLog   REQUIRED MODULE)

set(SOURCES
    SYSRAP_LOG.cc 
    PLOG.cc 
    SBacktrace.cc
    SFrame.cc
    SProc.cc
    SSys.cc
    SStr.cc
    SSeq.cc
    STranche.cc
    SAr.cc
    SArgs.cc
    SPPM.cc
    SVec.cc
    SMap.cc
    SDigest.cc
    SDirect.cc
    SArrayDigest.cc
    SLauncher.cc
    SRenderer.cc
    SCtrl.cc
    STimes.cc
    SLog.cc
    SBase36.cc
    SSortKV.cc
    SPairVec.cc
    md5.cc
    SPPM.cc
    SId.cc
    SGDML.cc
) 

set(HEADERS
    OPTICKS_LOG.hh

    SYSRAP_LOG.hh
    SYSRAP_API_EXPORT.hh
    SYSRAP_HEAD.hh
    SYSRAP_TAIL.hh
    PLOG.hh
    PLOG_INIT.hh  

    SProc.hh
    SBacktrace.hh
    SFrame.hh
    SSys.hh
    SStr.hh
    SSeq.hh
    STranche.hh
    SVec.hh
    SMap.hh
    SDigest.hh
    SDirect.hh
    SArrayDigest.hh
    SLauncher.hh
    SRenderer.hh
    SCtrl.hh
    STimes.hh
    SLog.hh
    SBase36.hh
    SSortKV.hh
    SPairVec.hh
    OpticksCSG.h
    OpticksCSGMask.h
    PlainFormatter.hh
    SAr.hh
    SArgs.hh
    S_freopen_redirect.hh
    SPPM.hh
    SId.hh
    SGDML.hh
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )

target_link_libraries( ${name} PUBLIC Opticks::PLog Opticks::OKConf )
if(UNIX AND NOT APPLE)
message(STATUS "adding ssl crypto for UNIX AND NOT APPLE")
target_link_libraries( ${name} PUBLIC ssl crypto )
endif()

set(SysRap_VERBOSE OFF)
if(SysRap_VERBOSE)
  echo_target_std(Opticks::OKConf) 
  #echo_target_std(${name}) 
endif()

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_compile_definitions( ${name} PUBLIC OPTICKS_SYSRAP )



bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)


