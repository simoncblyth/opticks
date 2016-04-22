find_library( OpenMeshRap_LIBRARY 
              NAMES OpenMeshRap
              PATHS ${OPTICKS_PREFIX}/lib )

set( OpenMeshRap_LIBRARIES 
             ${OpenMeshRap_LIBRARY} 
)

#set(OpenMeshRap_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/OpenMeshRap")
set(OpenMeshRap_INCLUDE_DIRS "${OPTICKS_HOME}/graphics/openmeshrap")
set(OpenMeshRap_DEFINITIONS "")

