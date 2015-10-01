
set(OpenMeshRap_PREFIX "$ENV{LOCAL_BASE}/env/graphics/openmeshrap")

find_library( OpenMeshRap_LIBRARY 
              NAMES OpenMeshRap
              PATHS ${OpenMeshRap_PREFIX}/lib )

set( OpenMeshRap_LIBRARIES 
             ${OpenMeshRap_LIBRARY} 
)

set(OpenMeshRap_INCLUDE_DIRS "${OpenMeshRap_PREFIX}/include")
set(OpenMeshRap_DEFINITIONS "")

