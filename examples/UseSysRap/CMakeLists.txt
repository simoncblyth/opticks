cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)

set(name UseSysRap)
project(${name})

#include(/usr/local/opticks/config/SysRapTargets.cmake) 
include(${OPTICKS_CONFIG_DIR}/SysRapTargets.cmake) 

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

add_executable(${name} ${name}.cc)

target_link_libraries(${name} Opticks::SysRap)

install(TARGETS ${name}  DESTINATION lib)




