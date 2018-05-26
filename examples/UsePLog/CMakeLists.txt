cmake_minimum_required(VERSION 3.5)
set(name UsePLog)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(PLog REQUIRED MODULE)

add_executable(${name} ${name}.cc)

target_link_libraries(${name} PUBLIC Opticks::PLog )

install(TARGETS ${name} DESTINATION lib)


