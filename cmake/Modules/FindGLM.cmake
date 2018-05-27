
set(GLM_MODULE "${CMAKE_CURRENT_LIST_FILE}")
set(GLM_VERBOSE OFF)

find_path(
    GLM_INCLUDE_DIR
    NAMES "glm/glm.hpp"
    PATHS "${CMAKE_INSTALL_PREFIX}/externals/glm/glm"
)

if(GLM_INCLUDE_DIR)
  set(GLM_FOUND "YES")
else()
  set(GLM_FOUND "NO")
endif()

if(GLM_VERBOSE OR NOT GLM_FOUND)
  message(STATUS "GLM_MODULE           : ${GLM_MODULE}")
  message(STATUS "CMAKE_INSTALL_PREFIX : ${CMAKE_INSTALL_PREFIX} ")
  message(STATUS "GLM_INCLUDE_DIR      : ${GLM_INCLUDE_DIR} ")
  message(STATUS "GLM_FOUND            : ${GLM_FOUND}")
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



