cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name YoctoGLRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(YoctoGL  REQUIRED CONFIG)
find_package(NPY     REQUIRED CONFIG) 
# NPY depends on SysRap and YoctoGL among others, so no need to repeat here


set(SOURCES
    YOG_LOG.cc

    YOG.cc
    YOGTF.cc

    YOGMaker.cc
    YOGMakerImpl.cc
    YOGGeometry.cc
)

SET(HEADERS

    YOG.hh
    YOGTF.hh

    YOGMaker.hh
    YOGMakerImpl.hh
    YOGGeometry.hh

    YOG_LOG.hh
    YOG_API_EXPORT.hh
    YOG_HEAD.hh
    YOG_TAIL.hh
)

add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_link_libraries( ${name} PUBLIC 
    Opticks::NPY
)

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

target_compile_definitions( ${name} PUBLIC OPTICKS_YOG ) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


