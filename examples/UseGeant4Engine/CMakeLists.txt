cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseGeant4Engine)
project(${name})
include(OpticksBuildOptions)

find_package(Geant4 CONFIG)   

if(Geant4_FOUND)
    message(STATUS "Geant4_FOUND   : ${Geant4_FOUND} ")
    message(STATUS "Geant4_VERSION : ${Geant4_VERSION} ")
    message(STATUS "Geant4_CXXSTD  : ${Geant4_CXXSTD} ")
    message(STATUS "Geant4_DEFINITIONS : ${Geant4_DEFINITIONS} ")
    message(STATUS "Geant4_LIBRARIES : ${Geant4_LIBRARIES} ")
    message(STATUS "Geant4_INCLUDE_DIRS : ${Geant4_INCLUDE_DIRS} ")
    message(STATUS "Geant4_builtin_clhep_FOUND :${Geant4_builtin_clhep_FOUND} ")
    message(STATUS "Geant4_system_clhep_FOUND :${Geant4_system_clhep_FOUND} ")
    message(STATUS "Geant4_system_clhep_ISGRANULAR : ${Geant4_system_clhep_ISGRANULAR} ") 
    message(STATUS "CLHEP_DIR : ${CLHEP_DIR} ")
    message(STATUS "CLHEP_INCLUDE_DIRS : ${CLHEP_INCLUDE_DIRS} ")

    #echo_target_std(Geant4)
    echo_pfx_vars(Geant4 "INCLUDE_DIR;INCLUDE_DIRS;LIBRARIES;DEFINITIONS")
else()
    message(STATUS "Geant4 not found")
endif()


add_definitions(${Geant4_DEFINITIONS} -DG4USE_STD11)
#add_compile_definitions(G4USE_STD11)
#add_compile_definitions(G4MULTITHREADED)


#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)


target_link_libraries(${name} PUBLIC ${Geant4_LIBRARIES})

target_include_directories( ${name} PUBLIC
   $<BUILD_INTERFACE:${Geant4_INCLUDE_DIR}>
)

install(TARGETS ${name}  DESTINATION lib)


