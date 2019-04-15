cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseXercesC)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(XercesC MODULE REQUIRED)   


message(STATUS "XercesC_FOUND        : ${XercesC_FOUND} ")
message(STATUS "XercesC_VERSION      : ${XercesC_VERSION} ")
message(STATUS "XercesC_INCLUDE_DIRS : ${XercesC_INCLUDE_DIRS} ")
message(STATUS "XercesC_LIBRARIES    : ${XercesC_LIBRARIES} ")
message(STATUS "XercesC_INCLUDE_DIR  : ${XercesC_INCLUDE_DIR} ")
message(STATUS "XercesC_LIBRARY      : ${XercesC_LIBRARIES} ")


#[=[
See UseOpticksXercesC 
#]=] 


