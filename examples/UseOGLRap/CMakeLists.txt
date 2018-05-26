cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOGLRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(OGLRap_VERBOSE ON)
find_package(OGLRap REQUIRED CONFIG)   

message(STATUS "OGLRap_CONFIG  : ${OGLRap_CONFIG} ")
message(STATUS "OGLRap_DIR     : ${OGLRap_DIR} ")

#echo_pfx_vars(OGLRap "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::OGLRap)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OGLRap )
install(TARGETS ${name}  DESTINATION lib)

