cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(STATUS " CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR} ")
   message(STATUS " CMAKE_BINARY_DIR : ${CMAKE_BINARY_DIR} ")
   message(FATAL_ERROR "in-source build detected : DONT DO THAT")
endif()

set(name Opticks)
project(${name} VERSION 0.1.0)

include(CTest)
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})


set(SUBDIRS
  okconf
  sysrap
  boostrap
  npy
  optickscore
  ggeo 
  assimprap
  openmeshrap 
  opticksgeo 
  cudarap
  thrustrap
  optixrap
  okop
  oglrap  
  opticksgl
  ok
  cfg4
  okg4
)

#[=[
Opticks CMake Naming Conventions 
==================================

1. project names are the same as the library targets they generate 
2. project names are CamelCased, project directory names are all lowercase
3. lowercased project names match the names of their directories

#]=]

macro(find_package)    
   set(pkg ${ARGV0}) 
   set(_arglist "${ARGV}")

   string(TOLOWER ${pkg} lpkg) 
   if(NOT "${lpkg}" IN_LIST SUBDIRS)
       message(STATUS "find_package ( ${ARGV} ) START ")
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


