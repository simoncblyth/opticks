cmake_minimum_required(VERSION 3.0)
set(name UseUseGLMViaBCM)
project(${name} VERSION 0.1.0)

set(dep UseGLMViaBCM)

find_package(${dep} 0.1.0 REQUIRED)

include(EchoTarget)
echo_target_std(${dep})

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable(${name} ${name}.cc)

target_link_libraries( ${name} PUBLIC UseGLMViaBCM)

install(TARGETS ${name} DESTINATION lib)



