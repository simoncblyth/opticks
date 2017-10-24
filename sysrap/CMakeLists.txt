cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name SysRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)
find_package(PLog REQUIRED)


set(LIBRARIES
   ${PLog_LIBRARIES}
)

if(UNIX) 
   set(LIBRARIES ${LIBRARIES} ssl crypto)
endif(UNIX)

message("${name}:CMAKE_BINARY_DIR : ${CMAKE_BINARY_DIR} ")
#message("${name}:CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS} ")
#message("${name}:LIBRARIES : ${LIBRARIES} ")
#message("${name}:PLog_INCLUDE_DIRS : ${PLog_INCLUDE_DIRS} ")

include_directories(
   ${CMAKE_BINARY_DIR}/inc         # top level inc for OpticksCMakeConfig.hh 
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}
   ${PLog_INCLUDE_DIRS} 
)

add_definitions(
    ${PLog_DEFINITIONS} 
)

set(SOURCES
    SYSRAP_LOG.cc 
    PLOG.cc 
    SProc.cc
    SSys.cc
    SSeq.cc
    SAr.cc
    SPPM.cc
    SVec.cc
    SDigest.cc
    SArrayDigest.cc
    SLauncher.cc
    SRenderer.cc
    STimes.cc
    SLog.cc
    SBase36.cc
    SSortKV.cc
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
    SVec.hh
    SDigest.hh
    SArrayDigest.hh
    SLauncher.hh
    SRenderer.hh
    STimes.hh
    SLog.hh
    SBase36.hh
    SSortKV.hh
    OpticksCSG.h
)

add_library( ${name}  SHARED ${SOURCES})
target_link_libraries( ${name}  ${LIBRARIES} )

install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)


