# http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries

set(GLEW_PREFIX "$ENV{LOCAL_BASE}/env/graphics/glew/1.12.0")

LINK_DIRECTORIES(${GLEW_PREFIX}/lib)

find_library( GLEW_LIBRARY 
              NAMES glew GLEW
              PATHS ${GLEW_PREFIX}/lib )

set( GLEW_LIBRARIES ${GLEW_LIBRARY} )

set(GLEW_INCLUDE_DIRS "${GLEW_PREFIX}/include")
set(GLEW_DEFINITIONS "")

