find_library( GGeoView_LIBRARIES 
              NAMES GGeoView
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT GGeoView_LIBRARIES)
       set(GGeoView_LIBRARIES GGeoView)
    endif()
endif(SUPERBUILD)


set(GGeoView_INCLUDE_DIRS "${GGeoView_SOURCE_DIR}")
set(GGeoView_DEFINITIONS "")

