#[=[
Idea is for this to find whatever OptiX 5,6 or 7 that is available in the CMake path 
and do the target setup in a way that hides the differences 
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





