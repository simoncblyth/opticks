cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseCUDARap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(CUDARap REQUIRED CONFIG)
#echo_target_std(Opticks::CUDARap)

add_executable( ${name} ${name}.cc )
target_link_libraries(${name} PUBLIC Opticks::CUDARap )
install(TARGETS ${name} DESTINATION lib)

