set(G4DAE_MODULE "${CMAKE_CURRENT_LIST_FILE}")
set(G4DAE_PREFIX "${CMAKE_INSTALL_PREFIX}/externals")

find_path(
    G4DAE_INCLUDE_DIR
    NAMES "G4DAEParser.hh"
    PATHS "${G4DAE_PREFIX}/include"
)

find_library( 
    G4DAE_LIBRARY 
    NAMES "G4DAE"
    PATHS "${G4DAE_PREFIX}/lib" 
)

if(G4DAE_INCLUDE_DIR AND G4DAE_LIBRARY)
    set(G4DAE_FOUND "YES")
else()
    set(G4DAE_FOUND "NO")
endif()


set(G4DAE_VERBOSE ON)
if(G4DAE_VERBOSE)
    message(STATUS "FindG4DAE.cmake G4DAE_PREFIX       : ${G4DAE_PREFIX}  ")
    message(STATUS "FindG4DAE.cmake G4DAE_INCLUDE_DIR  : ${G4DAE_INCLUDE_DIR}  ")
    message(STATUS "FindG4DAE.cmake G4DAE_LIBRARY      : ${G4DAE_LIBRARY}  ")
    message(STATUS "FindG4DAE.cmake G4DAE_FOUND        : ${G4DAE_FOUND}  ")
endif()

if(G4DAE_FOUND AND NOT TARGET Opticks::G4DAE)
    add_library(Opticks::G4DAE UNKNOWN IMPORTED) 
    set_target_properties(Opticks::G4DAE PROPERTIES
        IMPORTED_LOCATION "${G4DAE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${G4DAE_INCLUDE_DIR}"
    )
endif()


