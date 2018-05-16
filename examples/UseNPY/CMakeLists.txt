cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)

set(name UseNPY)

project(${name})

include(${OPTICKS_CONFIG_DIR}/SysRapTargets.cmake) 
include(${OPTICKS_CONFIG_DIR}/NPYTargets.cmake) 
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  


add_executable(${name} ${name}.cc)

#target_include_directories(${name} PUBLIC Opticks::NPY)

target_link_libraries(${name} Opticks::SysRap Opticks::NPY)

install(TARGETS ${name}  DESTINATION lib)




