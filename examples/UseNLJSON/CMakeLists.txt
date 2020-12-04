cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseNLJSON)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


set(NLJSON_VERBOSE ON)

find_package(NLJSON REQUIRED MODULE)

add_executable(${name} ${name}.cc)

target_link_libraries(${name} PUBLIC Opticks::NLJSON)

install(TARGETS ${name}  DESTINATION lib)



