cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseBoost)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

#set(Boost_DEBUG ON)
#set(Boost_NO_BOOST_CMAKE ON)
#find_package(Boost REQUIRED COMPONENTS filesystem)   



set(COMPONENTS system program_options filesystem regex )
find_package(Boost REQUIRED COMPONENTS ${COMPONENTS} )

#set_target_properties(Boost::system  PROPERTIES INTERFACE_FIND_PACKAGE_NAME "Boost REQUIRED COMPONENTS ${COMPONENTS}")
set_target_properties(Boost::system  PROPERTIES INTERFACE_FIND_PACKAGE_EXACT "REQUIRED COMPONENTS ${COMPONENTS}")

#[=[
setting INTERFACE_FIND_PACKAGE_NAME is a kludge (see bcm:share/bcm/cmake/BCMExport.cmake)

that tees up arguments to find_dependency in generated export useboost-config.cmake 
so downstream will automatically do the above find_package 

Found that with Boost 1.70.0 and the INTERFACE_FIND_PACKAGE_NAME kludge setting 
of "Boost REQUIRED COMPONENTS ${COMPONENTS}" that the version string gets stuffed on the 
end which is mis-interpreted as a component causing CMake errors.
The share/bcm/cmake/BCMExport.cmake generated useboost-config.cmake which is invoked by bcm_deploy
had version string in the wrong place::
 
   #find_dependency(Boost REQUIRED COMPONENTS system;program_options;filesystem;regex 1.70.0)
   find_dependency(Boost 1.70.0 REQUIRED COMPONENTS system;program_options;filesystem;regex)
 
Workaround is to set INTERFACE_FIND_PACKAGE_EXACT to "REQUIRED COMPONENTS ${COMPONENTS}"
which keeps the version string as the 2nd argument. 

#]=]


echo_pfx_vars(Boost "VERSION;DIR;FOUND;FILESYSTEM_FOUND;SYSTEM_FOUND;INCLUDE_DIRS;LIBRARY_DIRS;FIND_COMPONENTS")
echo_target_std(Boost::filesystem)

set(SOURCES ${name}.cc ) 
set(HEADERS ${name}.hh )

add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} PUBLIC 
   Boost::system
   Boost::program_options
   Boost::filesystem
   Boost::regex
)


#[=[
set(example PTreeMinimal)
add_executable(${example} ${example}.cc)
target_link_libraries( ${example} PUBLIC 
   Boost::system
   Boost::program_options
   Boost::filesystem
   Boost::regex
)
#]=]



set(BCM_VERBOSE ON)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL) 
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

echo_target_std(${name})


