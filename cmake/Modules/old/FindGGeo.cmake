find_library( GGeo_LIBRARIES 
              NAMES GGeo
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT GGeo_LIBRARIES)
       set(GGeo_LIBRARIES GGeo)
    endif()
endif(SUPERBUILD)


set(GGeo_INCLUDE_DIRS "${GGeo_SOURCE_DIR}")
set(GGeo_DEFINITIONS "")

