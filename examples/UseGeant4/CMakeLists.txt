cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseGeant4)
project(${name})
    
include(EchoTarget)

find_package(Geant4 CONFIG QUIET)   

if(Geant4_FOUND)
    message(STATUS "Geant4_FOUND   : ${Geant4_FOUND} ")
    message(STATUS "Geant4_VERSION : ${Geant4_VERSION} ")
    #echo_target_std(Geant4)
    echo_pfx_vars(Geant4 "INCLUDE_DIR;INCLUDE_DIRS;LIBRARIES;DEFINITIONS")
else()
    message(STATUS "Geant4 not found")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC ${Geant4_LIBRARIES})

target_include_directories( ${name} PUBLIC
   $<BUILD_INTERFACE:${Geant4_INCLUDE_DIR}>
)

install(TARGETS ${name}  DESTINATION lib)


