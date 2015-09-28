# http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries

set(OpenMesh_PREFIX "$ENV{LOCAL_BASE}/env/graphics/OpenMesh/4.1")

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

