cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(STATUS " CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR} ")
   message(STATUS " CMAKE_BINARY_DIR : ${CMAKE_BINARY_DIR} ")
   message(FATAL_ERROR "in-source build detected : DONT DO THAT")
endif()

set(name OpticksExamples)
project(${name} VERSION 0.1.0)

#[=[
OpticksExamples built in integrated fashion
==============================================

Using to debug the thrap-t integrated build issue::

    epsilon:examples blyth$ ThrustRapMinimal
    libc++abi.dylib: terminating with uncaught exception of type thrust::system::system_error: device free failed: CUDA driver version is insufficient for CUDA runtime version
    Abort trap: 6


Get the issue with::

    set(SUBDIRS
      cudarap
      thrustrap
      UseCUDARapThrust
    )

    set(SUBDIRS
      cudarap
      ThrustRapMinimal
    )

Do not see the issue with::

    set(SUBDIRS
      ThrustRapMinimal
    )

Fix appears to be removing the "kludge clear" in cudarap/CMakeLists.txt::

     56 CUDA_ADD_LIBRARY( ${name} ${SOURCES} )
     57 #set_target_properties(${name}  PROPERTIES INTERFACE_LINK_LIBRARIES "" ) # kludge clear
     58 target_link_libraries( ${name} ${LIBRARIES})
     59 target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

Observe:

* integrated building is sensitive to messing up target properties in one sub-proj that 
  are reused in a subsequent one

* proj-by-proj building is sensitive only to what got persisted in prior builds 



#]=]


set(SUBDIRS
  cudarap
  ThrustRapMinimal
)


macro(find_package)    
   set(pkg ${ARGV0}) 
   set(_arglist "${ARGV}")

   string(TOLOWER ${pkg} lpkg) 
   if(NOT "${lpkg}" IN_LIST SUBDIRS)
       #message(STATUS "find_package ( ${ARGV} ) START ")
       _find_package(${ARGV})   
       set(_pkg_found ${${pkg}_FOUND})

       message(STATUS "find_package ( ${ARGV} ) --> ${_pkg_found} ")
       
       if(NOT "QUIET" IN_LIST _arglist AND NOT ${_pkg_found})
           message(FATAL_ERROR "ABORT : failed to find non-QUIET pkg ${pkg} ")
       endif()
   else()
       message(STATUS "find_package ( ${ARGV} ) SKIP integrated pkg ")
   endif()
endmacro()


foreach(_dir ${SUBDIRS})
   message(STATUS "${_dir}")
   add_subdirectory(${_dir})
endforeach()


