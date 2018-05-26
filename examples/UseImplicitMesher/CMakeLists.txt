cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseImplicitMesher)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(ImplicitMesher REQUIRED CONFIG)   

message(STATUS "ImplicitMesher_DIR   : ${ImplicitMesher_DIR} ")
message(STATUS "ImplicitMesher_FOUND : ${ImplicitMesher_FOUND} ")

echo_target_std(Opticks::ImplicitMesher)

add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::ImplicitMesher)

install(TARGETS ${name}  DESTINATION lib)

