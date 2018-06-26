#[=[

*THIS CMakeLists.txt IS NO LONGER USED** 

Opticks CMake Conventions (STILL APPLY)
=========================================

1. project names are the same as the library targets they generate 
2. project names are CamelCased, project directory names are lowercase
3. lowercased project names match the names of their directories
4. every CMake project command should be followed by "include(OpticksBuildOptions)"
5. the name of every project should be set via a "name" variable : this is used within
   the OpticksBuildOptions to set the install location of headers as include/${name}

For example a project called DemoProj would be setup in demoproj subdirectory with
a CMakeLists.txt file starting::

    cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
    set(name DemoProj)
    project(${name} VERSION 0.1.0)
    include(OpticksBuildOptions)


DECISION TO HALT ATTEMPTS TO GET INTEGRATED BUILD OPERATIONAL AGAIN
---------------------------------------------------------------------

Because it doesnt add enough value for the effort, instead 
just use proj-by-proj integrated at bash level with om.bash, 
see om-install

This Integrated Build  : ON HOLD
------------------------------------

For this integrated build, find_package is redefined by the below
macro to skip packages corresponding to the below SUBDIRS as they have directly 
been built already within the single CMake context and hence do not need to be found.

This structure allows the proj-by-proj building and integrated building to otherwise
be identical.

bcm_ignore_package
~~~~~~~~~~~~~~~~~~~

Incase decide to try again with integrated build, try 
out the bcm_ignore_package approach:

* http://bcm.readthedocs.io/en/latest/src/Building.html#integrated-builds



#]=]

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Opticks)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(INTEGRATED_SUBDIRS
  okconf
  sysrap
#[=[
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
#]=]
)


macro(find_package)    
   set(_args "${ARGV}")
   set(_pkg ${ARGV0}) 
   string(TOLOWER ${_pkg} _subdir) 
   if(NOT "${_subdir}" IN_LIST INTEGRATED_SUBDIRS)
       _find_package(${ARGV})   
       set(_pkg_found ${${_pkg}_FOUND})
       if(NOT ${_pkg} STREQUAL "BCM")  
           message(STATUS "find_package ( ${ARGV} ) --> ${_pkg_found} ")
       endif()
       if(NOT "QUIET" IN_LIST _args AND NOT ${_pkg_found})
           message(FATAL_ERROR "ABORT : failed to find non-QUIET pkg ${_pkg} ")
       endif()
   else()
       #set(_tgt "Opticks::${_pkg}")
       #
       ## BEFORE so changes to headers in build directory take precedence 
       ## over any old installed headers 
       ##target_include_directories( ${_tgt} BEFORE INTERFACE  
       #    $<BUILD_INTERFACE:${${_pkg}_SOURCE_DIR}>
       #    $<BUILD_INTERFACE:${${_pkg}_BINARY_DIR}/inc>
       #) 
   endif()
endmacro()

foreach(_dir ${INTEGRATED_SUBDIRS})
   add_subdirectory(${_dir})
endforeach()

