cmake_minimum_required(VERSION 3.0)

set(name UsePLog)

project(${name})

find_package(PLog REQUIRED)

add_executable(${name} ${name}.cc)

target_link_libraries(${name} PUBLIC Opticks::PLog )

install(TARGETS ${name}  DESTINATION lib)


