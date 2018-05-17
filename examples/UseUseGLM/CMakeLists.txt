cmake_minimum_required(VERSION 3.0)
set(name UseUseGLM)
project(${name} VERSION 0.1.0)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

find_package(UseGLM 0.1.0 REQUIRED)

include(EchoTarget)
echo_target(Opticks::UseGLM "INTERFACE_INCLUDE_DIRECTORIES;INTERFACE_LINK_LIBRARIES;INTERFACE_FIND_PACKAGE_NAME;INTERFACE_FIND_PACKAGE_VERSION;INTERFACE_FIND_PACKAGE_EXACT")   
# must quote list variables when passing to function

add_executable(${name}  ${name}.cc)

target_link_libraries( ${name}
     PUBLIC
        Opticks::UseGLM
)

install(TARGETS ${name}  DESTINATION lib)

