
#[=[
/**
CMake function top_meta_target
=================================

* canonically used from sub-project CMakeLists.txt to 
  prepare the TOPMETA string in ini format which is included 
  as a CMake comment within the BCM installed top matter  

* the TOPMETA string is parsed by bin/bcm.py giving python access to: 

   * locations of libraries 
   * include directories 
   * Hmm : missing compilation definitions

* this was implemented in an attempt to allow building against the 
  Opticks release without using CMake, instead the as yet to be 
  written opticks-config.py can be used to access all 
  the same information


**/
#]=]


function(top_meta_target _out _name _tgts )

    set(_prps 
    INTERFACE_INCLUDE_DIRECTORIES
    INTERFACE_LINK_LIBRARIES
    INTERFACE_IMPORTED_LOCATION
    )

    #message(STATUS "top_meta_target _out ${_out} _tgts ${_tgts} prps ${_prps}" )

    set(_TOPMETA)
    string(APPEND _TOPMETA "#[=[ TOPMETA ${_name}\n\n") 

    foreach(tgt ${_tgts})
       set(qtgt "Opticks::${tgt}")    

       if(TARGET ${qtgt})
          #message(STATUS "qtgt ${qtgt}")
          string(APPEND _TOPMETA "[${qtgt}]\n") 
          foreach(prp ${_prps})
             get_property(val TARGET ${qtgt} PROPERTY ${prp} ) 
             if("${val}" STREQUAL "" )
             else() 
                 string(APPEND _TOPMETA "${prp}:${val}\n") 
                 #message(STATUS "${prp}:${val}")
             endif() 
          endforeach()
       elseif(TARGET ${tgt})   # allow unqualified for the OptiX targets
          #message(STATUS "tgt ${tgt}")
          string(APPEND _TOPMETA "[${tgt}]\n") 
          foreach(prp ${_prps})
             get_property(val TARGET ${tgt} PROPERTY ${prp} ) 
             if("${val}" STREQUAL "" )
             else() 
                 string(APPEND _TOPMETA "${prp}:${val}\n") 
                 #message(STATUS "${prp}:${val}")
             endif() 
          endforeach()
       else()
           message(STATUS "no target ${tgt} or ${qtgt}")
       endif()
    endforeach()

    #message(STATUS "top_meta_target:_TOPMETA:${_TOPMETA}") 
    string(APPEND _TOPMETA "\n#]=]\n") 

    set(${_out} ${_TOPMETA} PARENT_SCOPE)   ## critical to set result into the PARENT_SCOPE

endfunction()





