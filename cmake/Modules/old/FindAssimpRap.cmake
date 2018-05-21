find_library( AssimpRap_LIBRARIES 
              NAMES AssimpRap
              PATHS ${OPTICKS_PREFIX}/lib )


if(SUPERBUILD)
    if(NOT AssimpRap_LIBRARIES)
       set(AssimpRap_LIBRARIES AssimpRap)
    endif()
endif(SUPERBUILD)


set(AssimpRap_INCLUDE_DIRS "${AssimpRap_SOURCE_DIR}")
set(AssimpRap_DEFINITIONS "")

