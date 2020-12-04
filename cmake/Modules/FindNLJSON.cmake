set(NLJSON_MODULE "${CMAKE_CURRENT_LIST_FILE}")
#set(NLJSON_VERBOSE OFF)

find_path(
    NLJSON_INCLUDE_DIR
    NAMES "json.hpp"
    PATHS "${OPTICKS_PREFIX}/externals/include/nljson"
)

if(NLJSON_INCLUDE_DIR)
  set(NLJSON_FOUND "YES")
else()
  set(NLJSON_FOUND "NO")
endif()


if(NLJSON_VERBOSE OR NOT NLJSON_FOUND)
  message(STATUS "OPTICKS_PREFIX           : ${OPTICKS_PREFIX}")
  message(STATUS "NLJSON_MODULE            : ${NLJSON_MODULE}")
  message(STATUS "NLJSON_INCLUDE_DIR       : ${NLJSON_INCLUDE_DIR} ")
  message(STATUS "NLJSON_FOUND             : ${NLJSON_FOUND}")
endif()

if(NOT NLJSON_FOUND)
  message(FATAL_ERROR "NLJSON NOT FOUND") 
endif()

set(_tgt Opticks::NLJSON)
if(NLJSON_FOUND AND NOT TARGET ${_tgt})
    add_library(${_tgt} INTERFACE IMPORTED)
    set_target_properties(${_tgt} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NLJSON_INCLUDE_DIR}"
        INTERFACE_PKG_CONFIG_NAME "NLJSON"
    )
    set(NLJSON_targets "NLJSON")
endif()



