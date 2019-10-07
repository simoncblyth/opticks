
set(OpticksAssimp_MODULE "${CMAKE_CURRENT_LIST_FILE}")

#if(NOT OPTICKS_PREFIX)
#    # this works when this module is included from installed tree
#    get_filename_component(OpticksAssimp_MODULE_DIR ${CMAKE_CURRENT_LIST_FILE} DIRECTORY)
#    get_filename_component(OpticksAssimp_MODULE_DIRDIR ${OpticksAssimp_MODULE_DIR} DIRECTORY)
#    get_filename_component(OpticksAssimp_MODULE_DIRDIRDIR ${OpticksAssimp_MODULE_DIRDIR} DIRECTORY)
#    set(OPTICKS_PREFIX ${OpticksAssimp_MODULE_DIRDIRDIR})
#endif()

set(OpticksAssimp_PREFIX "${OPTICKS_PREFIX}/externals")


find_path(
    OpticksAssimp_INCLUDE_DIR
    NAMES "assimp/Importer.hpp"
    PATHS "${OpticksAssimp_PREFIX}/include"
)

find_library( 
    OpticksAssimp_LIBRARY
    NAMES assimp  assimp-vc100-mtd
    PATHS ${OpticksAssimp_PREFIX}/lib 
)


if(OpticksAssimp_INCLUDE_DIR AND OpticksAssimp_LIBRARY)
  set(OpticksAssimp_FOUND "YES")
else()
  set(OpticksAssimp_FOUND "NO")
endif()


#set(OpticksAssimp_VERBOSE OFF)
if(OpticksAssimp_VERBOSE)
   message(STATUS "FindOpticksAssimp.cmake OpticksAssimp_PREFIX:${OpticksAssimp_PREFIX}  ")
   message(STATUS "FindOpticksAssimp.cmake OpticksAssimp_INCLUDE_DIR:${OpticksAssimp_INCLUDE_DIR}  ")
   message(STATUS "FindOpticksAssimp.cmake OpticksAssimp_LIBRARY:${OpticksAssimp_LIBRARY}  ")
   message(STATUS "FindOpticksAssimp.cmake OpticksAssimp_FOUND:${OpticksAssimp_FOUND}  ")
endif()

if(OpticksAssimp_FOUND AND NOT TARGET Opticks::OpticksAssimp)

    add_library(Opticks::OpticksAssimp UNKNOWN IMPORTED) 
    set_target_properties(Opticks::OpticksAssimp PROPERTIES
        IMPORTED_LOCATION "${OpticksAssimp_LIBRARY}"
        INTERFACE_IMPORTED_LOCATION "${OpticksAssimp_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OpticksAssimp_INCLUDE_DIR}"
    )

    set(OpticksAssimp_targets OpticksAssimp)

endif()


