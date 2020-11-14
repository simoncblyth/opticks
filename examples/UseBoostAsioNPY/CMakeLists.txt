cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseBoostAsioNPY)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(BoostAsio MODULE)
message(STATUS "BoostAsio_INCLUDE_DIR : ${BoostAsio_INCLUDE_DIR} ")
message(STATUS "BoostAsio_FOUND       : ${BoostAsio_FOUND} ")

if(NOT BoostAsio_FOUND)
   message(FATAL_ERROR "The \"boost/asio.hpp\" header is required for this example. Try: boost-;boost-rebuild-with-asio " )
endif()

find_package(NPY REQUIRED CONFIG)


set(SOURCES
   npy_header.cc
)  
set(HEADERS 
   npy_header.hh
)


add_library(${name} SHARED ${SOURCES} ${HEADERS})

target_link_libraries(${name} Opticks::NPY)

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)


install(TARGETS ${name}  DESTINATION lib)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL) 
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

