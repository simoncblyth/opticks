cmake_minimum_required(VERSION 3.5)
set(name UseGLM)
project(${name})

include(GNUInstallDirs)
set(CMAKE_INSTALL_INCLUDEDIR "include/${name}")

find_package(BCM)
include(BCMDeploy)
include(BCMSetupVersion)

bcm_setup_version(VERSION 0.1.0 GENERATE_HEADER ${name}Version.hh)

find_package(GLM REQUIRED)

add_library(${name} SHARED ${name}.cc)

set(HEADERS ${name}.hh)

target_link_libraries(${name} PUBLIC Opticks::GLM )

bcm_deploy(TARGETS ${name} INCLUDE include NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})



