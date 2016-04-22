find_library( OpticksGL_LIBRARIES 
              NAMES OpticksGL
              PATHS ${OPTICKS_PREFIX}/lib )

#set(OpticksGL_INCLUDE_DIRS "${OPTICKS_PREFIX}/include/OpticksGL")
set(OpticksGL_INCLUDE_DIRS "${OPTICKS_HOME}/opticksgl")
set(OpticksGL_DEFINITIONS "")

