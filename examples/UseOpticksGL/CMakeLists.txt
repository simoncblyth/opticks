cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksGL)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()



set(OpticksGL_VERBOSE ON)
find_package(OpticksGL REQUIRED CONFIG)   

message(STATUS "OpticksGL_CONFIG  : ${OpticksGL_CONFIG} ")
message(STATUS "OpticksGL_DIR     : ${OpticksGL_DIR} ")

#echo_pfx_vars(OpticksGL "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::OpticksGL)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OpticksGL )
install(TARGETS ${name}  DESTINATION lib)

