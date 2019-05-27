cmake_minimum_required(VERSION 3.5)
set(name UseUseBoost)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)



#[=[

The BCM approach generally manages to prevent having to 
worry about dependencies of your dependencies.
But boost itself appears to be special 
perhaps because its imported targets are "foreign" ones, 
not being created by BCM, and not having the expected properties.

See ../README.rst for the kludge done upstream in UseBoost
that tries to do the below find automatically 
#]=]

set(COMPONENTS system program_options filesystem regex )
find_package(Boost REQUIRED COMPONENTS ${COMPONENTS} )
set_target_properties(Boost::system  PROPERTIES INTERFACE_FIND_PACKAGE_NAME "Boost REQUIRED COMPONENTS ${COMPONENTS}")


find_package(UseBoost 0.1.0 REQUIRED)

echo_target_std(Opticks::UseBoost)

add_executable(${name} ${name}.cc)
target_link_libraries( ${name} PUBLIC Opticks::UseBoost )

install(TARGETS ${name} DESTINATION lib)


