cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseG4OK)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

include(OpticksCMakeTraverse)



if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
    cmake_policy(SET CMP0077 OLD)
endif()


set(OpticksXercesC_VERBOSE ON)

find_package(G4OK CONFIG REQUIRED)   


#echo_pfx_vars(G4OK "FOUND;DIR;VERSION;VERSION_INTEGER;INCLUDE_DIR;INCLUDE_DIRS;LIBRARIES;DEFINITIONS")
echo_target(Opticks::G4OK "INTERFACE_LINK_LIBRARIES;INTERFACE_INCLUDE_DIRECTORIES;INTERFACE_COMPILE_DEFINITIONS" )
#echo_target_twolevel(Opticks::G4OK "INTERFACE_LINK_LIBRARIES" "IMPORTED_LOCATION")


add_executable(${name} ${name}.cc)
target_link_libraries(${name} PUBLIC Opticks::G4OK)

#install(TARGETS ${name}  DESTINATION ${CMAKE_INSTALL_LIBDIR})   this goes into lib64 which is not my standard
install(TARGETS ${name}  DESTINATION lib) 



set(LL_VERBOSE OFF)

set(PROP "LOCATION")
#set(PROP "INTERFACE_INCLUDE_DIRECTORIES")
#set(PROP "INTERFACE_COMPILE_DEFINITIONS")

set(OUT_LIST)
traverse_out(OUT_LIST ${PROP} Opticks::G4OK)
foreach(OUT ${OUT_LIST})
   message(STATUS ".... traverse_out.OUT: ${OUT} " )
endforeach()



