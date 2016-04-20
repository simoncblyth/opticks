
set(OpticksGL_PREFIX "$ENV{LOCAL_BASE}/env/opticksgl")

find_library( OpticksGL_LIBRARIES 
              NAMES OpticksGL
              PATHS ${OpticksGL_PREFIX}/lib )

set(OpticksGL_INCLUDE_DIRS "${OpticksGL_PREFIX}/include")
set(OpticksGL_DEFINITIONS "")

