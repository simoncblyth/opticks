cmake_minimum_required(VERSION 3.5)
set(name UseUseBoost)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


set(dep UseBoost)


# the below fixes it, but the whole point of the BCM approach
# is to avoid having to be concerned with the deps of your deps
# ... is this just a problem just for Boost ? Because its imported
# targets are "foreign" ones, not being created by BCM. 
#
#find_package(Boost REQUIRED COMPONENTS filesystem)
#find_package(Boost COMPONENTS filesystem)
#
# see ../README.rst for the kludge done upstream in UseBoost
# that enables this to be done automatically 
#

find_package(${dep} 0.1.0 REQUIRED)

echo_target_std(Opticks::${dep})

#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_executable(${name} ${name}.cc)
target_link_libraries( ${name} PUBLIC Opticks::${dep})


install(TARGETS ${name} DESTINATION lib)


