find_library( CUDARap_LIBRARIES 
              NAMES CUDARap
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT CUDARap_LIBRARIES)
       set(CUDARap_LIBRARIES CUDARap)
    endif()
endif(SUPERBUILD)


set(CUDARap_INCLUDE_DIRS "${CUDARap_SOURCE_DIR}")
set(CUDARap_DEFINITIONS "")

