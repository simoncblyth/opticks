
#set(OpenMesh_PREFIX "${OPTICKS_PREFIX}/externals/openmesh/4.1")
set(OpenMesh_PREFIX "${OPTICKS_PREFIX}/externals")

find_library( OpenMeshCore_LIBRARY 
              NAMES OpenMeshCored
              PATHS ${OpenMesh_PREFIX}/lib )

find_library( OpenMeshTools_LIBRARY 
              NAMES OpenMeshToolsd
              PATHS ${OpenMesh_PREFIX}/lib )


set( OpenMesh_LIBRARIES 
             ${OpenMeshCore_LIBRARY} 
             ${OpenMeshTools_LIBRARY} 
)

set(OpenMesh_INCLUDE_DIRS "${OpenMesh_PREFIX}/include")
set(OpenMesh_DEFINITIONS "")

