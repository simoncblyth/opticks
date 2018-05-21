cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpticksGeometry)
project(${name} VERSION 0.1.0)

find_package(OpticksGeometry REQUIRED CONFIG)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) 
add_executable(${name} ${name}.cc)
target_link_libraries(${name} Opticks::OpticksGeometry)
install(TARGETS ${name}  DESTINATION lib)

