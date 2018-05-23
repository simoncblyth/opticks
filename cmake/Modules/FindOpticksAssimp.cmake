
set(OpticksAssimp_PREFIX "${CMAKE_INSTALL_PREFIX}/externals")

find_path(
    OpticksAssimp_INCLUDE_DIR
    NAMES "assimp/Importer.hpp"
    PATHS "${OpticksAssimp_PREFIX}/include"
)

find_library( OpticksAssimp_LIBRARY
              NAMES assimp  assimp-vc100-mtd
              PATHS ${OpticksAssimp_PREFIX}/lib )



if(OpticksAssimp_INCLUDE_DIR AND OpticksAssimp_LIBRARY)
set(OpticksAssimp_FOUND "YES")
else()
set(OpticksAssimp_FOUND "NO")
endif()


set(OpticksAssimp_DEBUG OFF)
if(OpticksAssimp_DEBUG)
   message(STATUS "FindOpticksAssimp.cmake OpticksAssimp_PREFIX:${OpticksAssimp_PREFIX}  ")
   message(STATUS "FindOpticksAssimp.cmake OpticksAssimp_INCLUDE_DIR:${OpticksAssimp_INCLUDE_DIR}  ")
   message(STATUS "FindOpticksAssimp.cmake OpticksAssimp_LIBRARY:${OpticksAssimp_LIBRARY}  ")
   message(STATUS "FindOpticksAssimp.cmake OpticksAssimp_FOUND:${OpticksAssimp_FOUND}  ")
endif()

if(OpticksAssimp_FOUND AND NOT TARGET Opticks::OpticksAssimp)

    add_library(Opticks::OpticksAssimp UNKNOWN IMPORTED) 
    set_target_properties(Opticks::OpticksAssimp PROPERTIES
        IMPORTED_LOCATION "${OpticksAssimp_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OpticksAssimp_INCLUDE_DIR}"
    )

endif()


