cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name SysRap)
project(${name} VERSION 0.1.0)

include(EnvBuildOptions)
include(EnvCompilationFlags)

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)

find_package(PLog REQUIRED)


set(SOURCES
    SYSRAP_LOG.cc 
    PLOG.cc 
    SProc.cc
    SSys.cc
    SSeq.cc
    STranche.cc
    SAr.cc
    SPPM.cc
    SVec.cc
    SMap.cc
    SDigest.cc
    SArrayDigest.cc
    SLauncher.cc
    SRenderer.cc
    STimes.cc
    SLog.cc
    SBase36.cc
    SSortKV.cc
    SPairVec.cc
    md5.cc
) 

set(HEADERS
    SYSRAP_LOG.hh
    SYSRAP_API_EXPORT.hh
    SYSRAP_HEAD.hh
    SYSRAP_TAIL.hh
    PLOG.hh
    PLOG_INIT.hh  
    SProc.hh
    SSys.hh
    SSeq.hh
    SVec.hh
    SMap.hh
    SDigest.hh
    SArrayDigest.hh
    SLauncher.hh
    SRenderer.hh
    STimes.hh
    SLog.hh
    SBase36.hh
    SSortKV.hh
    SPairVec.hh
    OpticksCSG.h
    PlainFormatter.hh
    SAr.hh
)


add_library( ${name}  SHARED ${SOURCES})

target_link_libraries( ${name} PUBLIC Opticks::PLog )
if(UNIX)
target_link_libraries( ${name} PUBLIC ssl crypto )
endif()


bcm_deploy(TARGETS ${name} INCLUDE include NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


#add_subdirectory(tests EXCLUDE_FROM_ALL)
#add_subdirectory(tests)


