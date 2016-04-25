# http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries

set(GLFW_PREFIX "${OPTICKS_PREFIX}/externals/glfw/glfw")

find_library( GLFW_LIBRARY 
              NAMES glfw3 glfw
              PATHS ${GLFW_PREFIX}/lib )

set( GLFW_LIBRARIES ${GLFW_LIBRARY} )

if(APPLE)
    find_library( Cocoa_LIBRARY NAMES Cocoa )
    find_library( OpenGL_LIBRARY NAMES OpenGL )
    find_library( IOKit_LIBRARY NAMES IOKit )
    find_library( CoreFoundation_LIBRARY NAMES CoreFoundation )
    find_library( CoreVideo_LIBRARY NAMES CoreVideo )

    set( GLFW_LIBRARIES 
               ${GLFW_LIBRARIES} 
               ${Cocoa_LIBRARY}
               ${OpenGL_LIBRARY}
               ${IOKit_LIBRARY}
               ${CoreFoundation_LIBRARY} 
               ${CoreVideo_LIBRARY} )
endif(APPLE)

set(GLFW_INCLUDE_DIRS "${GLFW_PREFIX}/include")
set(GLFW_DEFINITIONS "")

