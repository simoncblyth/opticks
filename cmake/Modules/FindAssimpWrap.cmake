find_library( AssimpWrap_LIBRARIES 
              NAMES AssimpWrap
              PATHS ${OPTICKS_PREFIX}/lib )


if(SUPERBUILD)
    if(NOT AssimpWrap_LIBRARIES)
       set(AssimpWrap_LIBRARIES AssimpWrap)
    endif()
endif(SUPERBUILD)



set(AssimpWrap_INCLUDE_DIRS "${AssimpWrap_SOURCE_DIR}")
set(AssimpWrap_DEFINITIONS "")

