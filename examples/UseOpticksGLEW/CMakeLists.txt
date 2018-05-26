cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksGLEW)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)
    

set(OpticksGLEW_VERBOSE ON)
find_package(OpticksGLEW REQUIRED MODULE)   

message(STATUS "OpticksGLEW_DIR     : ${OpticksGLEW_DIR} ")  
message(STATUS "OpticksGLEW_VERSION : ${OpticksGLEW_VERSION} ")
message(STATUS "OpticksGLEW_CONFIG  : ${OpticksGLEW_CONFIG} ")

echo_pfx_vars(OpticksGLEW "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::OpticksGLEW)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OpticksGLEW )
install(TARGETS ${name}  DESTINATION lib)


