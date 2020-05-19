cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseCLHEP)
project(${name})
include(OpticksBuildOptions)

find_package(CLHEP CONFIG)   

if(CLHEP_FOUND)
    message(STATUS "CLHEP_DIR : ${CLHEP_DIR} ")
    message(STATUS "CLHEP_INCLUDE_DIRS : ${CLHEP_INCLUDE_DIRS} ")
    message(STATUS "CLHEP_LIBRARIES : ${CLHEP_LIBRARIES} ")
else()
    message(STATUS "CLHEP not found")
endif()


add_executable(${name} ${name}.cc)

target_link_libraries(${name} PUBLIC CLHEP::CLHEP)

install(TARGETS ${name}  DESTINATION lib)


