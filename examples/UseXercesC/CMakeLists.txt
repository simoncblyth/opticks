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


set(tgt Example::XercesC)
if(XercesC_FOUND AND NOT TARGET ${tgt})
    add_library(${tgt} UNKNOWN IMPORTED) 
    set_target_properties(${tgt} 
         PROPERTIES 
            IMPORTED_LOCATION             "${XercesC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${XercesC_INCLUDE_DIR}"
            INTERFACE_PKG_CONFIG_NAME     "xerces-c"
    )   
endif()

#[=[
See UseOpticksXercesC which tries to use the Geant4 XercesC and 
automates the setup of the imported target
#]=] 

message(STATUS "XercesC_MODULE  : ${XercesC_MODULE} ")

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC ${tgt} )
install(TARGETS ${name}  DESTINATION lib)

