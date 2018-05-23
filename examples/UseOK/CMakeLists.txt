cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOK)
project(${name} VERSION 0.1.0)

set(OK_VERBOSE ON)
find_package(OK REQUIRED CONFIG)   

message(STATUS "OK_CONFIG  : ${OK_CONFIG} ")
message(STATUS "OK_DIR     : ${OK_DIR} ")

include(EchoTarget)
#echo_pfx_vars(OK "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::OK)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OK )
install(TARGETS ${name}  DESTINATION lib)

