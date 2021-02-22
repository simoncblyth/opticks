cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseGGeoLoad)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(GGeo REQUIRED CONFIG)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} Opticks::GGeo)
install(TARGETS ${name}  DESTINATION lib)

