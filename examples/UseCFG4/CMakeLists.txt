cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseCFG4)
project(${name} VERSION 0.1.0)

set(CFG4_VERBOSE ON)
find_package(CFG4 REQUIRED CONFIG)   

message(STATUS "CFG4_CONFIG  : ${CFG4_CONFIG} ")
message(STATUS "CFG4_DIR     : ${CFG4_DIR} ")

include(EchoTarget)
#echo_pfx_vars(CFG4 "FOUND;VERSION;CONFIG;INCLUDE;LIBRARY;DLL;INSTALL_DIR")
echo_target_std(Opticks::CFG4)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::CFG4 )
install(TARGETS ${name}  DESTINATION lib)

