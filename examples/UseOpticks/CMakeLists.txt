cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticks)
project(${name} VERSION 0.1.0)

find_package(OpticksNonExisting QUIET)
message(STATUS "OpticksNonExisting_FOUND  : ${OpticksNonExisting_FOUND} ")


set(Opticks_VERBOSE ON)
find_package(Opticks QUIET)

message(STATUS "Opticks_FOUND   : ${Opticks_FOUND} ")
message(STATUS "Opticks_MODULE  : ${Opticks_MODULE} ")



