cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksCore)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(OpticksCore REQUIRED CONFIG)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # pulls in libs from both PREFIX/lib and PREFIX/externals/lib 
add_executable(${name} ${name}.cc)
target_link_libraries(${name} Opticks::OpticksCore)
install(TARGETS ${name}  DESTINATION lib)

