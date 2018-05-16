cmake_minimum_required(VERSION 3.5)
set(name UseGLMViaBCM)
project(${name})

find_package(BCM)
include(BCMDeploy)
include(BCMSetupVersion)

bcm_setup_version(VERSION 0.1.0 GENERATE_HEADER ${name}Version.hh)

find_package(GLM REQUIRED)

set(SOURCES
   ${name}.cc
)
add_library(${name} SHARED ${SOURCES})

target_link_libraries( ${name} PUBLIC
    GLM::GLM
)

bcm_deploy(TARGETS ${name} INCLUDE include)

