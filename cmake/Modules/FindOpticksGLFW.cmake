
set(OpticksGLFW_MODULE "${CMAKE_CURRENT_LIST_FILE}")

if(NOT OPTICKS_PREFIX)
    # this works when this module is included from installed tree
    get_filename_component(OpticksGLFW_MODULE_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
    get_filename_component(OpticksGLFW_MODULE_DIRDIR ${OpticksGLFW_MODULE_DIR} DIRECTORY)
    get_filename_component(OpticksGLFW_MODULE_DIRDIRDIR ${OpticksGLFW_MODULE_DIRDIR} DIRECTORY)
    set(OPTICKS_PREFIX ${OpticksGLFW_MODULE_DIRDIRDIR})
endif()


set(OpticksGLFW_PREFIX "${OPTICKS_PREFIX}/externals")
set(OpticksGLFW_ALT "ON")


find_path( OpticksGLFW_INCLUDE_DIR
           NAMES GLFW/glfw3.h
           PATHS 
              /usr/include
              ${OpticksGLFW_PREFIX}/include 
         )

# remove glfw3 as causes to find system static lib
find_library( OpticksGLFW_LIBRARY 
              NAMES glfw glfw3dll
              PATHS 
                   /usr/lib64
                   ${OpticksGLFW_PREFIX}/lib
              )

if(OpticksGLFW_INCLUDE_DIR AND OpticksGLFW_LIBRARY)
   set(OpticksGLFW_FOUND "YES")
else()
   set(OpticksGLFW_FOUND "NO")
endif()

#[=[
#]=]

if(OpticksGLFW_FOUND AND NOT TARGET Opticks::OpticksGLFW)
    set(tgt Opticks::OpticksGLFW)
    add_library(${tgt} UNKNOWN IMPORTED) 
    set_target_properties(${tgt} PROPERTIES IMPORTED_LOCATION "${OpticksGLFW_LIBRARY}")

    if(APPLE)
       find_library( Cocoa_FRAMEWORK NAMES Cocoa )
       find_library( OpenGL_FRAMEWORK NAMES OpenGL )
       find_library( IOKit_FRAMEWORK NAMES IOKit )
       find_library( CoreFoundation_FRAMEWORK NAMES CoreFoundation )
       find_library( CoreVideo_FRAMEWORK NAMES CoreVideo )

       if(OpticksGLFW_ALT)
           message(STATUS "FindOpticksGLFW.cmake : ALT " )
           set_target_properties(${tgt} PROPERTIES 
                      INTERFACE_LINK_LIBRARIES 
                         "${Cocoa_FRAMEWORK};${OpenGL_FRAMEWORK};${IOKit_FRAMEWORK};${CoreFoundation_FRAMEWORK};${CoreVideo_FRAMEWORK}")
 
       else()
           message(STATUS "FindOpticksGLFW.cmake : not-ALT " )
           target_link_libraries(${tgt} INTERFACE 
               ${Cocoa_FRAMEWORK}
               ${OpenGL_FRAMEWORK}
               ${IOKit_FRAMEWORK} 
               ${CoreFoundation_FRAMEWORK}
               ${CoreVideo_FRAMEWORK}
          )
      endif()

#[=[
NB cannot just use "-framework Cocoa" etc, theres some secret distinguishing frameworks apparently 
#]=]

      if(OpticksGLFW_VERBOSE)
          message(STATUS "FindOpticksGLFW.cmake : Cocoa_FRAMEWORK : ${Cocoa_FRAMEWORK}")
          message(STATUS "FindOpticksGLFW.cmake : OpenGL_FRAMEWORK : ${OpenGL_FRAMEWORK}")
          message(STATUS "FindOpticksGLFW.cmake : IOKit_FRAMEWORK : ${IOKit_FRAMEWORK}")
          message(STATUS "FindOpticksGLFW.cmake : CoreFoundation_FRAMEWORK : ${CoreFoundation_FRAMEWORK}")
          message(STATUS "FindOpticksGLFW.cmake : CoreVideo_FRAMEWORK : ${CoreVideo_FRAMEWORK}")
      endif()

    else()
       if(OpticksGLFW_ALT)
           set_target_properties(${tgt} PROPERTIES INTERFACE_LINK_LIBRARIES GL)
       else() 
           target_link_libraries(${tgt} INTERFACE GL)
       endif()
 
    endif()

    set_target_properties(${tgt} PROPERTIES 
        INTERFACE_INCLUDE_DIRECTORIES "${OpticksGLFW_INCLUDE_DIR}" 
        INTERFACE_FIND_PACKAGE_NAME "OpticksGLFW MODULE REQUIRED"
    )

    ## Above target_properties INTERFACE_FIND_PACKAGE_NAME kludge tees up the arguments 
    ## to find_dependency in BCM generated exports 
    ## so downstream targets will automatically do the required find_dependency
    ## and call this script again to revive the targets.
    ## NB INTERFACE_FIND_PACKAGE_NAME is a BCM defined property, not a standard one, see bcm-

endif()

if(OpticksGLFW_VERBOSE)
  message(STATUS "FindOpticksGLFW.cmake : OpticksGLFW_ALT         : ${OpticksGLFW_ALT} " )
  message(STATUS "FindOpticksGLFW.cmake : OpticksGLFW_MODULE      : ${OpticksGLFW_MODULE} " )
  message(STATUS "FindOpticksGLFW.cmake : OpticksGLFW_LIBRARY     : ${OpticksGLFW_LIBRARY} " )
  message(STATUS "FindOpticksGLFW.cmake : OpticksGLFW_INCLUDE_DIR : ${OpticksGLFW_INCLUDE_DIR} " )
endif()


