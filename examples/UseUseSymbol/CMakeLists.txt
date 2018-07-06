cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseUseSymbol)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(UseSymbol 0.1.0 REQUIRED)

add_executable(${name} ${name}.cc)

target_link_libraries( ${name} PUBLIC Opticks::UseSymbol)

install(TARGETS ${name} DESTINATION lib)


