# http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries

set(GLEW_PREFIX "$ENV{LOCAL_BASE}/env/graphics/glew/1.12.0")

find_library( GLEW_LIBRARY 
              NAMES glew
              PATHS ${GLEW_PREFIX}/lib )

set( GLEW_LIBRARIES ${GLEW_LIBRARY} )

set(GLEW_INCLUDE_DIRS "${GLEW_PREFIX}/include")
set(GLEW_DEFINITIONS "")

