cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseAssimpRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OpticksAssimp_VERBOSE ON)

find_package(AssimpRap REQUIRED CONFIG)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) 
add_executable(${name}Test ${name}.cc)
target_link_libraries(${name}Test Opticks::AssimpRap)
install(TARGETS ${name}Test  DESTINATION lib)

