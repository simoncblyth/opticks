

set(GLM_MODULE "${CMAKE_CURRENT_LIST_FILE}")
set(GLM_VERBOSE OFF)

#if(NOT OPTICKS_PREFIX)
#    # this works when this module is included from installed tree
#    get_filename_component(GLM_MODULE_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
#    get_filename_component(GLM_MODULE_DIRDIR ${GLM_MODULE_DIR} DIRECTORY)
#    get_filename_component(GLM_MODULE_DIRDIRDIR ${GLM_MODULE_DIRDIR} DIRECTORY)
#    set(OPTICKS_PREFIX ${GLM_MODULE_DIRDIRDIR})
#endif()

find_path(
    GLM_INCLUDE_DIR
    NAMES "glm/glm.hpp"
    PATHS "${OPTICKS_PREFIX}/externals/glm/glm"
)

if(GLM_INCLUDE_DIR)
  set(GLM_FOUND "YES")
else()
  set(GLM_FOUND "NO")
endif()


if(GLM_VERBOSE OR NOT GLM_FOUND)
  message(STATUS "OPTICKS_PREFIX           : ${OPTICKS_PREFIX}")
  message(STATUS "GLM_MODULE_DIR           : ${GLM_MODULE_DIR}")
  message(STATUS "GLM_MODULE_DIRDIR        : ${GLM_MODULE_DIRDIR}")
  message(STATUS "GLM_MODULE_DIRDIRDIR     : ${GLM_MODULE_DIRDIRDIR}")
  message(STATUS "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "GLM_MODULE               : ${GLM_MODULE}")
  message(STATUS "GLM_INCLUDE_DIR          : ${GLM_INCLUDE_DIR} ")
  message(STATUS "GLM_FOUND                : ${GLM_FOUND}")
endif()


if(NOT GLM_FOUND)
  message(FATAL_ERROR "GLM NOT FOUND") 
endif()


set(_tgt Opticks::GLM)
if(GLM_FOUND AND NOT TARGET ${_tgt})
    add_library(${_tgt} INTERFACE IMPORTED)
    set_target_properties(${_tgt} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIR}"
    )
endif()



