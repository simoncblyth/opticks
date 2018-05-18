cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpenMesh)
project(${name})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)  

find_package(OpenMesh REQUIRED MODULE)   

message(STATUS "OpenMesh_DIR : ${OpenMesh_DIR} ")
message(STATUS "OpenMesh_FOUND : ${OpenMesh_FOUND} ")

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::OpenMesh)

install(TARGETS ${name}  DESTINATION lib)

