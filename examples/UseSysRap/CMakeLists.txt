cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

set(name UseSysRap)

project(${name})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(SysRap REQUIRED CONFIG)   
## CONFIG option loads the targets, preventing  
## use of the old variable based cmake/Modules/FindSysRap.cmake

message(STATUS "SysRap_DIR : ${SysRap_DIR} ")

add_executable(${name} ${name}.cc)

target_link_libraries(${name} Opticks::SysRap)

install(TARGETS ${name}  DESTINATION lib)

