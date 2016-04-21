
set(OpticksGL_PREFIX "${OPTICKS_PREFIX}/opticksgl")

find_library( OpticksGL_LIBRARIES 
              NAMES OpticksGL
              PATHS ${OpticksGL_PREFIX}/lib )

set(OpticksGL_INCLUDE_DIRS "${OpticksGL_PREFIX}/include")
set(OpticksGL_DEFINITIONS "")

