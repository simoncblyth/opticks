#[=[

testing the snippet to find Opticks with in::

   $JUNOTOP/offline/cmake/JUNODependencies.cmake

#]=]

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticks)
project(${name} VERSION 0.1.0)

find_package(OpticksNonExisting QUIET)
message(STATUS "OpticksNonExisting_FOUND  : ${OpticksNonExisting_FOUND} ")


## Opticks
if(DEFINED ENV{OPTICKS_PREFIX})
   set(Opticks_VERBOSE YES)
   set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{OPTICKS_HOME}/cmake/Modules")
   find_package(Opticks QUIET MODULE)
   message(STATUS "${CMAKE_CURRENT_LIST_FILE} : Opticks_FOUND:${Opticks_FOUND}" )
endif()





