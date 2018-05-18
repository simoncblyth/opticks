cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseDualContouringSample)
project(${name})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(DualContouringSample REQUIRED CONFIG)   

message(STATUS "DualContouringSample_DIR   : ${DualContouringSample_DIR} ")
message(STATUS "DualContouringSample_FOUND : ${DualContouringSample_FOUND} ")

include(EchoTarget)
echo_target_std(Opticks::DualContouringSample)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::DualContouringSample)

install(TARGETS ${name}  DESTINATION lib)

