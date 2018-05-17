
#set(PLog_PREFIX "${OPTICKS_PREFIX}/externals/plog")
#set(PLog_LIBRARIES "")
#set(PLog_INCLUDE_DIRS "${PLog_PREFIX}/include")
#set(PLog_DEFINITIONS "")

find_path(
    PLog_INCLUDE_DIR 
    NAMES "plog/Log.h"
    PATHS "${CMAKE_INSTALL_PREFIX}/externals/plog/include"
)


if(PLog_INCLUDE_DIR)
   set(PLog_FOUND "YES")
else()
   set(PLog_FOUND "NO")
endif()


if(PLog_FOUND AND NOT TARGET Opticks::PLog)

    add_library(Opticks::PLog INTERFACE IMPORTED)
    set_target_properties(Opticks::PLog PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PLog_INCLUDE_DIR}"
    )

else()
    message("FindPLog ${PLog_FOUND} failed to FindPLog")
endif()



