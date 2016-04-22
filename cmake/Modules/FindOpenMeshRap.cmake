find_library( OpenMeshRap_LIBRARIES
              NAMES OpenMeshRap
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OpenMeshRap_LIBRARIES)
       set(OpenMeshRap_LIBRARIES OpenMeshRap)
    endif()
endif(SUPERBUILD)


set(OpenMeshRap_INCLUDE_DIRS "${OpenMeshRap_SOURCE_DIR}")
set(OpenMeshRap_DEFINITIONS "")

