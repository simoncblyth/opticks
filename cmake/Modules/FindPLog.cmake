
set(PLog_MODULE ${CMAKE_CURRENT_LIST_FILE})

message(STATUS "PLog_MODULE : ${PLog_MODULE}" )


#set(PLog_VERBOSE OFF)

#[=[
Hmm tis kinda awkward for the externals to be inside the prefix when 
handling multiple versions of opticks that want to share externals 
#]=]


find_path(
    PLog_INCLUDE_DIR 
    NAMES "plog/Log.h"
    PATHS "${OPTICKS_PREFIX}/externals/plog/include"
)


if(PLog_INCLUDE_DIR)
   set(PLog_FOUND "YES")
else()
   set(PLog_FOUND "NO")
endif()

set(_tgt Opticks::PLog)

set(PLog_targets)

if(PLog_FOUND AND NOT TARGET ${_tgt})

    add_library(${_tgt} INTERFACE IMPORTED)
    set_target_properties(${_tgt} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PLog_INCLUDE_DIR}"
    )
    list(APPEND PLog_targets "PLog")

endif()

if(PLog_VERBOSE)
    message(STATUS "FindPLog.cmake : PLog_MODULE      : ${PLog_MODULE} ")
    message(STATUS "FindPLog.cmake : OPTICKS_PREFIX   : ${OPTICKS_PREFIX} ")
    message(STATUS "FindPLog.cmake : PLog_INCLUDE_DIR : ${PLog_INCLUDE_DIR} ")
    message(STATUS "FindPLog.cmake : PLog_FOUND       : ${PLog_FOUND}  ")
endif()


