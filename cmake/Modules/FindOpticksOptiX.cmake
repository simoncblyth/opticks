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
   set(OptiX_VERSION_INTEGER ${OpticksOptiX_VERSION}) # needed by OKConf machinery 
endif()


if(OpticksOptiX_VERBOSE)
    if(OpticksOptiX_VERSION LESS 70000)
        message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OpticksOptiX_VERSION : ${OpticksOptiX_VERSION} : is pre-7  ")
    else()
        message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OpticksOptiX_VERSION : ${OpticksOptiX_VERSION} : is 7+ ")
    endif()
endif() 




if(OpticksOptiX_FOUND AND OpticksOptiX_VERSION LESS 70000)

    if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT APPLE)
      set(bit_dest "64")
    else()
      set(bit_dest "")
    endif()

    macro(OPTIX_find_api_library name version)
      find_library(${name}_LIBRARY
        NAMES ${name}.${version} ${name}
        PATHS "${OptiX_INSTALL_DIR}/lib${bit_dest}"
        NO_DEFAULT_PATH
        )
      find_library(${name}_LIBRARY
        NAMES ${name}.${version} ${name}
        )
      if(WIN32)
        find_file(${name}_DLL
          NAMES ${name}.${version}.dll
          PATHS "${OptiX_INSTALL_DIR}/bin${bit_dest}"
          NO_DEFAULT_PATH
          )
        find_file(${name}_DLL
          NAMES ${name}.${version}.dll
          )
      endif()
    endmacro()

    OPTIX_find_api_library(optix 1)
    OPTIX_find_api_library(optixu 1)
    OPTIX_find_api_library(optix_prime 1)

    if(OpticksOptiX_VERBOSE)
        message(STATUS "${CMAKE_CURRENT_LIST_FILE} : optix_LIBRARY       : ${optix_LIBRARY} ")
        message(STATUS "${CMAKE_CURRENT_LIST_FILE} : optixu_LIBRARY      : ${optixu_LIBRARY} ")
        message(STATUS "${CMAKE_CURRENT_LIST_FILE} : optix_prime_LIBRARY : ${optix_prime_LIBRARY} ")
    endif() 


    # Macro for setting up dummy targets
    function(OptiX_add_imported_library name lib_location dll_lib dependent_libs)
      set(CMAKE_IMPORT_FILE_VERSION 1)

      add_library(${name} SHARED IMPORTED)

      if(WIN32)
        set_target_properties(${name} PROPERTIES
          IMPORTED_IMPLIB "${lib_location}"
          IMPORTED_LOCATION "${dll_lib}"
          IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
          INTERFACE_IMPORTED_LOCATION "${lib_location}"        
          INTERFACE_IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
          )
      elseif(UNIX)
        set_target_properties(${name} PROPERTIES
          IMPORTED_LOCATION "${lib_location}"
          IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
          INTERFACE_IMPORTED_LOCATION "${lib_location}"          
          INTERFACE_IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
          )
      else()
        set_target_properties(${name} PROPERTIES
          IMPORTED_LOCATION "${lib_location}"
          IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
          INTERFACE_IMPORTED_LOCATION "${lib_location}"       
          INTERFACE_IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
          )
      endif()
      set(CMAKE_IMPORT_FILE_VERSION)
    endfunction()

    OptiX_add_imported_library(optix         "${optix_LIBRARY}"       "${optix_DLL}"        "${OPENGL_LIBRARIES}")
    OptiX_add_imported_library(optixu        "${optixu_LIBRARY}"      "${optixu_DLL}"       "")
    OptiX_add_imported_library(optix_prime   "${optix_prime_LIBRARY}" "${optix_prime_DLL}"  "")

endif() 





if(OpticksOptiX_FOUND)
    add_library(Opticks::OptiX INTERFACE IMPORTED) 
    set_target_properties(Opticks::OptiX PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OpticksOptiX_INCLUDE}"
        INTERFACE_PKG_CONFIG_NAME "OptiX"
    )

    if(OpticksOptiX_VERSION LESS 70000)
        set_target_properties(Opticks::OptiX PROPERTIES
            INTERFACE_LINK_LIBRARIES "optix;optixu;optix_prime"
        )
    endif()


endif()





