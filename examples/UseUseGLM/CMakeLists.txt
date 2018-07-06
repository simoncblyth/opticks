cmake_minimum_required(VERSION 3.5)
set(name UseUseGLM)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(dep UseGLM)
find_package(${dep} 0.1.0 REQUIRED)

include(EchoTarget)
echo_target_std(Opticks::${dep})

#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable(${name} ${name}.cc)

target_link_libraries( ${name} PUBLIC Opticks::${dep})

install(TARGETS ${name} DESTINATION lib)



