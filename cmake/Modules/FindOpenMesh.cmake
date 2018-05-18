

set(OpenMesh_PREFIX "${CMAKE_INSTALL_PREFIX}/externals")

#set(OpenMesh_VERSION "3.4")
#set(OpenMesh_SUFFIX "")

## For SUSE Linux without debug libs.
#if(UNIX AND NOT APPLE)
#   set(OpenMesh_SUFFIX "")
#else(UNIX AND NOT APPLE)
#   set(OpenMesh_SUFFIX "")
#endif(UNIX AND NOT APPLE)
#
#  Release libs : OpenMeshCore  OpenMeshTools
#  Debug libs :   OpenMeshCored OpenMeshToolsd


find_path( OpenMesh_INCLUDE_DIR
           NAMES "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
           PATHS "${OpenMesh_PREFIX}/include"
)

find_library( OpenMeshCore_LIBRARY 
              NAMES OpenMeshCore${OpenMesh_SUFFIX}
              PATHS ${OpenMesh_PREFIX}/lib )

find_library( OpenMeshTools_LIBRARY 
              NAMES OpenMeshTools${OpenMesh_SUFFIX}
              PATHS ${OpenMesh_PREFIX}/lib )


#set( OpenMesh_LIBRARIES 
#             ${OpenMeshCore_LIBRARY} 
#             ${OpenMeshTools_LIBRARY} 
#)
#set(OpenMesh_INCLUDE_DIRS "${OpenMesh_PREFIX}/include")
#set(OpenMesh_DEFINITIONS "")

if(OpenMesh_INCLUDE_DIR AND OpenMeshCore_LIBRARY AND OpenMeshTools_LIBRARY)
set(OpenMesh_FOUND "YES")
else()
set(OpenMesh_FOUND "NO")
endif()

message(STATUS "FindOpenMesh.cmake OpenMesh_INCLUDE_DIR:${OpenMesh_INCLUDE_DIR}  ")
message(STATUS "FindOpenMesh.cmake OpenMeshCore_LIBRARY:${OpenMeshCore_LIBRARY}  ")
message(STATUS "FindOpenMesh.cmake OpenMeshTools_LIBRARY:${OpenMeshTools_LIBRARY}  ")
message(STATUS "FindOpenMesh.cmake OpenMesh_FOUND:${OpenMesh_FOUND}  ")



if(OpenMesh_FOUND AND NOT TARGET Opticks::OpenMesh)

    add_library(Opticks::OpenMeshCore UNKNOWN IMPORTED) 
    set_target_properties(Opticks::OpenMeshCore PROPERTIES
        IMPORTED_LOCATION "${OpenMeshCore_LIBRARY}"
    )

    add_library(Opticks::OpenMeshTools  UNKNOWN IMPORTED) 
    set_target_properties(Opticks::OpenMeshTools PROPERTIES
        IMPORTED_LOCATION "${OpenMeshTools_LIBRARY}"
    )

    add_library(Opticks::OpenMesh INTERFACE IMPORTED)
    set_target_properties(Opticks::OpenMesh PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OpenMesh_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "Opticks::OpenMeshCore;Opticks::OpenMeshTools"
    )

endif()


