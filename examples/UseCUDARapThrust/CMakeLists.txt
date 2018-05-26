cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseCUDARapThrust)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(CUDARap REQUIRED CONFIG)

echo_target_std(Opticks::CUDARap)

cuda_add_executable( ${name} ${name}.cu )

target_link_libraries(${name} Opticks::CUDARap )

install(TARGETS ${name} DESTINATION lib)

