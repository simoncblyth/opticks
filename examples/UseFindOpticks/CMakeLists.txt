cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseFindOpticks)
project(${name} VERSION 0.1.0)

if(DEFINED ENV{OPTICKS_PREFIX})
    set(Opticks_VERBOSE YES)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "$ENV{HOME}/opticks/cmake/Modules")
    find_package(Opticks MODULE)
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : Opticks_FOUND:${Opticks_FOUND}" )
else()
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : NOT Looking for Opticks AS OPTICKS_PREFIX NOT DEFINED")
endif()

