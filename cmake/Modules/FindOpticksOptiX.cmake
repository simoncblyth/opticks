#[=[
Idea is for this to find whatever OptiX 5,6 or 7 that is available in the CMake path 
and do the target setup in a way that hides the differences 

But the setup is more involved pre7 , so also changed FindOptiX to work with 7 
.. so situation not settled

#]=]

set(OpticksOptiX_MODULE  "${CMAKE_CURRENT_LIST_FILE}")


find_path(OpticksOptiX_INCLUDE
  NAMES optix.h
  PATHS "$ENV{OPTICKS_OPTIX_PREFIX}/include"
  NO_DEFAULT_PATH
  )


if(OpticksOptiX_VERBOSE)
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OpticksOptiX_VERBOSE : ${OpticksOptiX_VERBOSE} ")
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OpticksOptiX_MODULE  : ${OpticksOptiX_MODULE} ")
    message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OpticksOptiX_INCLUDE : ${OpticksOptiX_INCLUDE} ")
endif()



if(OpticksOptiX_INCLUDE)
   set(OpticksOptiX_FOUND "YES")   # hmm for 7 this is fine as no libs, but not pre7
else()
   set(OpticksOptiX_FOUND "NO")
endif()


set(OpticksOptiX_VERSION 0)
if(OpticksOptiX_FOUND)
   file(READ "${OpticksOptiX_INCLUDE}/optix.h" _contents)
   string(REGEX REPLACE "\n" ";" _contents "${_contents}")
   foreach(_line ${_contents})
        if (_line MATCHES "#define OPTIX_VERSION ([0-9]+)")  # no space after it with 7
            set(OpticksOptiX_VERSION ${CMAKE_MATCH_1} )
            #message(STATUS "FindOpticksOptiX.cmake._line ${_line} ===> ${CMAKE_MATCH_1} ") 
            #else()
            #message(STATUS "${_line}")
        endif()
   endforeach()
endif()


if(OpticksOptiX_FOUND)
  add_library(Opticks::OptiX INTERFACE IMPORTED) 
   set_target_properties(Opticks::OptiX PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OpticksOptiX_INCLUDE}"
        INTERFACE_PKG_CONFIG_NAME "OptiX"
   )
endif()





