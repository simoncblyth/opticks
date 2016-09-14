find_library( OKOP_LIBRARIES 
              NAMES OKOP
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OKOP_LIBRARIES)
       set(OKOP_LIBRARIES OKOP)
    endif()
endif(SUPERBUILD)


set(OKOP_INCLUDE_DIRS "${OKOP_SOURCE_DIR}")
set(OKOP_DEFINITIONS "")

