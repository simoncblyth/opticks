
set(OpticksGLEW_MODULE "${CMAKE_CURRENT_LIST_FILE}")

if(NOT OPTICKS_PREFIX)
    # this works when this module is included from installed tree
    get_filename_component(OpticksGLEW_MODULE_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
    get_filename_component(OpticksGLEW_MODULE_DIRDIR ${OpticksGLEW_MODULE_DIR} DIRECTORY)
    get_filename_component(OpticksGLEW_MODULE_DIRDIRDIR ${OpticksGLEW_MODULE_DIRDIR} DIRECTORY)
    set(OPTICKS_PREFIX ${OpticksGLEW_MODULE_DIRDIRDIR})
endif()

set(OpticksGLEW_PREFIX "${OPTICKS_PREFIX}/externals")

find_path( OpticksGLEW_INCLUDE_DIR
           NAMES "GL/glew.h"
           PATHS "${OpticksGLEW_PREFIX}/include"
)
find_library( OpticksGLEW_LIBRARY 
              NAMES glew GLEW libglew32 glew32
              PATHS ${OpticksGLEW_PREFIX}/lib )

if(OpticksGLEW_VERBOSE)
  message(STATUS "OpticksGLEW_MODULE      : ${OpticksGLEW_MODULE}")
  message(STATUS "OpticksGLEW_FOUND       : ${OpticksGLEW_FOUND}")
  message(STATUS "OpticksGLEW_PREFIX      : ${OpticksGLEW_PREFIX}")
  message(STATUS "OpticksGLEW_INCLUDE_DIR : ${OpticksGLEW_INCLUDE_DIR}")
  message(STATUS "OpticksGLEW_LIBRARY     : ${OpticksGLEW_LIBRARY}")
endif()

if(OpticksGLEW_INCLUDE_DIR AND OpticksGLEW_LIBRARY)
  set(OpticksGLEW_FOUND "YES")
else()
  set(OpticksGLEW_FOUND "NO")
endif()


if(OpticksGLEW_FOUND AND NOT TARGET Opticks::OpticksGLEW)
    add_library(Opticks::OpticksGLEW UNKNOWN IMPORTED) 
    set_target_properties(Opticks::OpticksGLEW PROPERTIES
        IMPORTED_LOCATION "${OpticksGLEW_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OpticksGLEW_INCLUDE_DIR}"
    )
endif()

