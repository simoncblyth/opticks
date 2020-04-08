cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name SysRap)
set(desc "System Level Utilities depending")
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
    SPath.cc
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
    S_get_option.cc
    STime.cc
    SASCII.cc
    SAbbrev.cc
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
    SPath.hh
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
    S_get_option.hh
    STime.hh
    SPPM.hh
    SId.hh
    SGDML.hh
    SASCII.hh
    SAbbrev.hh
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
#set_target_properties( ${name} 
#                       PROPERTIES 
#                         INTERFACE_DESCRIPTION ${desc}
#                         INTERFACE_PKG_CONFIG_REQUIRES ${preq}
#                     )

set_property( TARGET ${name} PROPERTY INTERFACE_DESCRIPTION ${desc} )
#set_property( TARGET ${name} PROPERTY INTERFACE_PKG_CONFIG_REQUIRES ${preq} ) 


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


# <findname>_targets is my convention
message(STATUS "PLog_targets:${PLog_targets}")

## top_meta_target(TOPMETA..)  function call from cmake/Modules/TopMetaTarget.cmake creates TOPMETA string into this scope
top_meta_target(PLog_TOPMETA    "PLog" "${PLog_targets}" )

message(STATUS "${PLog_TOPMETA}")
 
set(PC_VERBOSE ON)
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL TOPMATTER  "
## SysRap TOPMATTER

${PLog_TOPMETA}

## end SysRap TOPMATTER
")

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


set( SCRIPTS 
    __init__.py
    OpticksCSG.py
)
install(PROGRAMS ${SCRIPTS}  DESTINATION py/opticks/sysrap)


add_subdirectory(tests)



