cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseNPY)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(NPY REQUIRED CONFIG)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} Opticks::NPY)
install(TARGETS ${name}  DESTINATION lib)

