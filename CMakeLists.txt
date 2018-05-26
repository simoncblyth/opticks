#[=[
Opticks CMake Conventions 
==================================

1. project names are the same as the library targets they generate 
2. project names are CamelCased, project directory names are lowercase
3. lowercased project names match the names of their directories
4. every CMake project command should be followed by "include(OpticksBuildOptions)"

#]=]

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Opticks)
project(${name} VERSION 0.1.0)
set(FLAGS_VERBOSE ON)
include(OpticksBuildOptions)   
set(FLAGS_VERBOSE OFF)

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
       #message(STATUS "find_package ( ${ARGV} ) SKIP integrated pkg ")
   endif()
endmacro()


foreach(_dir ${SUBDIRS})
   add_subdirectory(${_dir})
endforeach()


