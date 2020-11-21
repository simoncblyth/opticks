cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseBoostAsioUDP)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(COMPONENTS system program_options filesystem regex)
find_package(Boost REQUIRED COMPONENTS ${COMPONENTS} )


#set(BoostAsio_VERBOSE ON)
find_package(BoostAsio MODULE)
message(STATUS "BoostAsio_INCLUDE_DIR : ${BoostAsio_INCLUDE_DIR} ")
message(STATUS "BoostAsio_FOUND       : ${BoostAsio_FOUND} ")

if(NOT BoostAsio_FOUND)
   message(FATAL_ERROR "The \"boost/asio.hpp\" header is required for this example. Try: boost-;boost-rebuild-with-asio " )
endif()


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)





set_target_properties(Boost::system  PROPERTIES INTERFACE_FIND_PACKAGE_EXACT "REQUIRED COMPONENTS ${COMPONENTS}") # see ../UseBoost/CMakeLists.txt 

set(SOURCES
    ListenUDP.cc
    MockViz.cc
)  
set(HEADERS 
    ListenUDP.hh
    MockViz.hh
)

add_library(${name} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${name} PUBLIC 
   Boost::system
   Boost::program_options
   Boost::filesystem
   Boost::regex
   Threads::Threads
)

target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)


bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL) 
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

