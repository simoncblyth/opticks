
#set(OpenMesh_PREFIX "${OPTICKS_PREFIX}/externals/openmesh/4.1")
set(OpenMesh_PREFIX "${OPTICKS_PREFIX}/externals")


## TODO: find better way to make this choice
if(WIN32)
   set(OpenMesh_SUFFIX "")
else(WIN32)
   set(OpenMesh_SUFFIX "d")
endif()
## For SUSE Linux without debug libs.
if(UNIX AND NOT APPLE)
   set(OpenMesh_SUFFIX "")
endif(UNIX AND NOT APPLE)

#  Release libs : OpenMeshCore  OpenMeshTools
#  Debug libs :   OpenMeshCored OpenMeshToolsd


find_library( OpenMeshCore_LIBRARY 
              NAMES OpenMeshCore${OpenMesh_SUFFIX}
              PATHS ${OpenMesh_PREFIX}/lib )

find_library( OpenMeshTools_LIBRARY 
              NAMES OpenMeshTools${OpenMesh_SUFFIX}
              PATHS ${OpenMesh_PREFIX}/lib )


set( OpenMesh_LIBRARIES 
             ${OpenMeshCore_LIBRARY} 
             ${OpenMeshTools_LIBRARY} 
)

set(OpenMesh_INCLUDE_DIRS "${OpenMesh_PREFIX}/include")
set(OpenMesh_DEFINITIONS "")

