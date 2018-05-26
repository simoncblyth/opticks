cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksAssimp)
project(${name})
include(OpticksBuildOptions)


find_package(OpticksAssimp REQUIRED MODULE)   
message(STATUS "OpticksAssimp_FOUND : ${OpticksAssimp_FOUND} ")

echo_target_std(Opticks::OpticksAssimp)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OpticksAssimp)

install(TARGETS ${name}  DESTINATION lib)

