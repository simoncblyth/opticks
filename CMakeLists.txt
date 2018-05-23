cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
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
)

## opticks convention : lowercased project names match directory names
macro(find_package)    
   string(TOLOWER ${ARGV0} lpkg) 
   if(NOT "${lpkg}" IN_LIST SUBDIRS)
       message(STATUS "find_package ( ${ARGV} )  proceed as ${lpkg} is not in subdirs list ")
       _find_package(${ARGV})   
   else()
       message(STATUS "find_package ( ${ARGV} )  SKIPPING as ${lpkg} is in subdirs list ")
   endif()
endmacro()


foreach(_dir ${SUBDIRS})
   message(STATUS "${_dir}")
   add_subdirectory(${_dir})
endforeach()


