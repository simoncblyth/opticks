
set(OpticksGLEW_PREFIX "${OPTICKS_PREFIX}/externals")

message("OpticksGLEW_PREFIX:${OpticksGLEW_PREFIX}")

LINK_DIRECTORIES(${OpticksGLEW_PREFIX}/lib)

find_library( OpticksGLEW_LIBRARY 
              NAMES glew GLEW libglew32 glew32
              PATHS ${OpticksGLEW_PREFIX}/lib )

set( OpticksGLEW_LIBRARIES ${OpticksGLEW_LIBRARY} )

set(OpticksGLEW_INCLUDE_DIRS "${OpticksGLEW_PREFIX}/include")
set(OpticksGLEW_DEFINITIONS "")

