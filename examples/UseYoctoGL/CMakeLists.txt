cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseYoctoGL)
project(${name})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(YoctoGL REQUIRED CONFIG)   

message(STATUS "YoctoGL_DIR   : ${YoctoGL_DIR} ")
message(STATUS "YoctoGL_FOUND : ${YoctoGL_FOUND} ")
include(EchoTarget)
echo_target_std(Opticks::YoctoGL)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::YoctoGL)

install(TARGETS ${name}  DESTINATION lib)

