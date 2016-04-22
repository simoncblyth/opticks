find_library( OpticksGL_LIBRARIES 
              NAMES OpticksGL
              PATHS ${OPTICKS_PREFIX}/lib )

if(SUPERBUILD)
    if(NOT OpticksGL_LIBRARIES)
       set(OpticksGL_LIBRARIES OpticksGL)
    endif()
endif(SUPERBUILD)

set(OpticksGL_INCLUDE_DIRS "${OpticksGL_SOURCE_DIR}")
set(OpticksGL_DEFINITIONS "")

