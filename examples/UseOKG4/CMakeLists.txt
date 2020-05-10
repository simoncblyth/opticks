cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOKG4)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


set(OKG4_VERBOSE ON)
find_package(OKG4 REQUIRED CONFIG)   
message(STATUS "OKG4_CONFIG  : ${OKG4_CONFIG} ")
message(STATUS "OKG4_DIR     : ${OKG4_DIR} ")

#echo_pfx_vars(OKG4 "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::OKG4)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OKG4 )
install(TARGETS ${name}  DESTINATION lib)

