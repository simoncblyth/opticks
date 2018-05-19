cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseCSGBSP)
project(${name})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(CSGBSP REQUIRED CONFIG)   

message(STATUS "CSGBSP_DIR   : ${CSGBSP_DIR} ")
message(STATUS "CSGBSP_FOUND : ${CSGBSP_FOUND} ")
include(EchoTarget)
echo_target_std(Opticks::CSGBSP)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::CSGBSP)

install(TARGETS ${name}  DESTINATION lib)

