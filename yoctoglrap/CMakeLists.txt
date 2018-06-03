cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name YoctoGLRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(SysRap  REQUIRED CONFIG) 
find_package(NPY     REQUIRED CONFIG) 
find_package(YoctoGL REQUIRED CONFIG)  # from /usr/local/opticks/externals/lib/cmake/yoctogl/  

set(YoctoGL_VERBOSE OFF)
if(YoctoGL_VERBOSE)
    message(STATUS "YoctoGL_DIR   : ${YoctoGL_DIR} ")
    message(STATUS "YoctoGL_FOUND : ${YoctoGL_FOUND} ")
    echo_target_std(Opticks::YoctoGL)
endif()

set(SOURCES
    YOG_LOG.cc

    YOGMaker.cc
    YOGGeometry.cc
)

SET(HEADERS
    YOGMaker.hh
    YOGGeometry.hh

    YOG_LOG.hh
    YOG_API_EXPORT.hh
    YOG_HEAD.hh
    YOG_TAIL.hh
)

add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_link_libraries( ${name} PUBLIC 
    Opticks::SysRap
    Opticks::NPY
    Opticks::YoctoGL
)

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
)

target_compile_definitions( ${name} PUBLIC OPTICKS_YOG ) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


