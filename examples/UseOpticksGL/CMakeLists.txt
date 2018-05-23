cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksGL)
project(${name} VERSION 0.1.0)

set(OpticksGL_VERBOSE ON)
find_package(OpticksGL REQUIRED CONFIG)   

message(STATUS "OpticksGL_CONFIG  : ${OpticksGL_CONFIG} ")
message(STATUS "OpticksGL_DIR     : ${OpticksGL_DIR} ")

include(EchoTarget)
#echo_pfx_vars(OpticksGL "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::OpticksGL)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OpticksGL )
install(TARGETS ${name}  DESTINATION lib)

