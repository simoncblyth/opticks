cmake_minimum_required(VERSION 3.5)
set(name UseGLM)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

bcm_setup_version(VERSION 0.1.0 GENERATE_HEADER ${name}Version.hh)

find_package(GLM REQUIRED MODULE)

add_library(${name} SHARED ${name}.cc)

set(HEADERS ${name}.hh)

target_link_libraries(${name} PUBLIC Opticks::GLM )

bcm_deploy(TARGETS ${name} INCLUDE include NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})



