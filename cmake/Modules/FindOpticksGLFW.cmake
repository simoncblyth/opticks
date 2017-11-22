

set(OpticksGLFW_PREFIX "${OPTICKS_PREFIX}/externals")

# remove glfw3 as causes to find system static lib
find_library( OpticksGLFW_LIBRARY 
              NAMES glfw glfw3dll
              PATHS ${OpticksGLFW_PREFIX}/lib )

set( OpticksGLFW_LIBRARIES ${OpticksGLFW_LIBRARY} )

if(APPLE)
    find_library( Cocoa_LIBRARY NAMES Cocoa )
    find_library( OpenGL_LIBRARY NAMES OpenGL )
    find_library( IOKit_LIBRARY NAMES IOKit )
    find_library( CoreFoundation_LIBRARY NAMES CoreFoundation )
    find_library( CoreVideo_LIBRARY NAMES CoreVideo )

    set( OpticksGLFW_LIBRARIES 
               ${OpticksGLFW_LIBRARIES} 
               ${Cocoa_LIBRARY}
               ${OpenGL_LIBRARY}
               ${IOKit_LIBRARY}
               ${CoreFoundation_LIBRARY} 
               ${CoreVideo_LIBRARY} )
endif(APPLE)

set(OpticksGLFW_INCLUDE_DIRS "${OpticksGLFW_PREFIX}/include")
set(OpticksGLFW_DEFINITIONS "")

#message("${name}.OpticksGLFW_PREFIX    : ${OpticksGLFW_PREFIX} ")
#message("${name}.OpticksGLFW_LIBRARIES : ${OpticksGLFW_LIBRARIES} ")


