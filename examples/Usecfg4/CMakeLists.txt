cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name Usecfg4)
project(${name} VERSION 0.1.0)

set(cfg4_VERBOSE ON)
find_package(cfg4 REQUIRED CONFIG)   

message(STATUS "cfg4_CONFIG  : ${cfg4_CONFIG} ")
message(STATUS "cfg4_DIR     : ${cfg4_DIR} ")

include(EchoTarget)
#echo_pfx_vars(cfg4 "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::cfg4)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::cfg4 )
install(TARGETS ${name}  DESTINATION lib)

