cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseG4)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(G4_VERBOSE ON)
find_package(G4 MODULE REQUIRED)   

#echo_pfx_vars(G4 "FOUND;DIR;VERSION;VERSION_INTEGER;INCLUDE_DIR;INCLUDE_DIRS;LIBRARIES;DEFINITIONS")
echo_target(Opticks::G4 "INTERFACE_LINK_LIBRARIES;INTERFACE_INCLUDE_DIRECTORIES;INTERFACE_COMPILE_DEFINITIONS" )
#echo_target_twolevel(Opticks::G4 "INTERFACE_LINK_LIBRARIES" "IMPORTED_LOCATION")


#echo_target( G4persistency "INTERFACE_LINK_LIBRARIES;INTERFACE_INCLUDE_DIRECTORIES;INTERFACE_COMPILE_DEFINITIONS" )



#[=[
# fishing for xercesc lib and include_dir : using interface link libraries of G4persistency
#set(xercesc_lib)
#set(xercesc_include_dir)

#get_target_property(_lll G4persistency INTERFACE_LINK_LIBRARIES)
#message(STATUS "_lll ${_lll} " )
#foreach(_lib ${_lll})
#    get_filename_component(_nam ${_lib} NAME) 
#    string(FIND "${_nam}" "libxerces-c" _pos ) 
#    if(_pos EQUAL 0)
        #message(STATUS "_lib ${_lib}  _nam ${_nam} _pos ${_pos} ") 
#        set(xercesc_lib ${_lib})
#    endif()
#endforeach()
#
#if(xercesc_lib)
#    get_filename_component(_dir ${xercesc_lib} DIRECTORY) 
#    get_filename_component(_dirdir ${_dir} DIRECTORY) 
#    set(xercesc_include_dir "${_dirdir}/include" )     # assumes include_dir is ../include relative to directory of the lib 
#endif()
#
#
#message(STATUS " xercesc_lib         : ${xercesc_lib} ")
#message(STATUS " xercesc_include_dir : ${xercesc_include_dir} ")
#
#]=]




set(SOURCES ${name}.cc)
set(HEADERS ${name}.hh)

add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} PUBLIC Opticks::G4)
#install(TARGETS ${name}  DESTINATION ${CMAKE_INSTALL_LIBDIR})   # bcm_deploy does this : no need to repeat
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)



