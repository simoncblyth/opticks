cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseBoost)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(Boost_DEBUG ON)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED COMPONENTS filesystem)   

## kludge that tees up arguments to find_dependency in generated export useboost-config.cmake 
## so downstream will automatically do the above find_package 
set_target_properties(Boost::filesystem PROPERTIES INTERFACE_FIND_PACKAGE_NAME "Boost REQUIRED COMPONENTS filesystem")

echo_pfx_vars(Boost "VERSION;DIR;FOUND;FILESYSTEM_FOUND;SYSTEM_FOUND;INCLUDE_DIRS;LIBRARY_DIRS;FIND_COMPONENTS")
echo_target_std(Boost::filesystem)

set(SOURCES ${name}.cc ) 
set(HEADERS ${name}.hh )

add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} PUBLIC Boost::filesystem)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL) 
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

echo_target_std(${name})


