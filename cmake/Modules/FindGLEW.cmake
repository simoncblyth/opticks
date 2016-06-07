# http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries

#set(GLEW_PREFIX "${OPTICKS_PREFIX}/externals/glew/glew")
set(GLEW_PREFIX "${OPTICKS_PREFIX}/externals")


#message("GLEW_PREFIX:${GLEW_PREFIX}")


LINK_DIRECTORIES(${GLEW_PREFIX}/lib)

find_library( GLEW_LIBRARY 
              NAMES glew GLEW libglew32
              PATHS ${GLEW_PREFIX}/lib )

set( GLEW_LIBRARIES ${GLEW_LIBRARY} )

set(GLEW_INCLUDE_DIRS "${GLEW_PREFIX}/include")
set(GLEW_DEFINITIONS "")

