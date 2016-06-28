
#set(OpenMesh_PREFIX "${OPTICKS_PREFIX}/externals/openmesh/4.1")
set(OpenMesh_PREFIX "${OPTICKS_PREFIX}/externals")


#
#  Release libs : OpenMeshCore  OpenMeshTools
#  Debug libs :   OpenMeshCored OpenMeshToolsd
#


find_library( OpenMeshCore_LIBRARY 
              NAMES OpenMeshCore
              PATHS ${OpenMesh_PREFIX}/lib )

find_library( OpenMeshTools_LIBRARY 
              NAMES OpenMeshTools
              PATHS ${OpenMesh_PREFIX}/lib )


set( OpenMesh_LIBRARIES 
             ${OpenMeshCore_LIBRARY} 
             ${OpenMeshTools_LIBRARY} 
)

set(OpenMesh_INCLUDE_DIRS "${OpenMesh_PREFIX}/include")
set(OpenMesh_DEFINITIONS "")

