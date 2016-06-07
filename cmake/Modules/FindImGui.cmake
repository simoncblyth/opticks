#set(ImGui_PREFIX "${OPTICKS_PREFIX}/externals/imgui/imgui.install")
set(ImGui_PREFIX "${OPTICKS_PREFIX}/externals")

find_library( ImGui_LIBRARY 
              NAMES ImGui 
              PATHS ${ImGui_PREFIX}/lib )

set( ImGui_LIBRARIES ${ImGui_LIBRARY} )

if(APPLE)
    find_library( Cocoa_LIBRARY NAMES Cocoa )
    find_library( OpenGL_LIBRARY NAMES OpenGL )
    find_library( IOKit_LIBRARY NAMES IOKit )
    find_library( CoreFoundation_LIBRARY NAMES CoreFoundation )
    find_library( CoreVideo_LIBRARY NAMES CoreVideo )

    set( ImGui_LIBRARIES 
               ${ImGui_LIBRARIES} 
               ${GLFW_LIBRARIES} 
               ${Cocoa_LIBRARY}
               ${OpenGL_LIBRARY}
               ${IOKit_LIBRARY}
               ${CoreFoundation_LIBRARY} 
               ${CoreVideo_LIBRARY} )
   #message("FindImGUI:APPLE")
endif(APPLE)

if(UNIX AND NOT APPLE)
    find_library( OpenGL_LIBRARY NAMES GL )

    set( ImGui_LIBRARIES 
               ${ImGui_LIBRARIES} 
               ${GLFW_LIBRARIES} 
               ${OpenGL_LIBRARY}
               )

   #message("FindImGUI:UNIX AND NOT APPLE")
endif(UNIX AND NOT APPLE)


if(MINGW)
    find_library( OpenGL_LIBRARY NAMES opengl32 )

    message("FindImGui MINGW OpenGL_LIBRARY : ${OpenGL_LIBRARY} " )

    set( ImGui_LIBRARIES 
               ${ImGui_LIBRARIES} 
               ${GLFW_LIBRARIES} 
               ${OpenGL_LIBRARY}
               )


endif(MINGW)





set(ImGui_INCLUDE_DIRS "${ImGui_PREFIX}/include")
set(ImGui_DEFINITIONS "")

