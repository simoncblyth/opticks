cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
# this file is copied into place by imgui-cmake-ize
set(name ImGui)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

set(BUILD_SHARED_LIBS ON)


find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)


set(LIBRARIES
   ${GLEW_LIBRARIES} 
   ${GLFW_LIBRARIES}
)

message("GLEW_LIBRARIES:${GLEW_LIBRARIES}")
message("GLFW_LIBRARIES:${GLFW_LIBRARIES}")


if(APPLE)
    message("APPLE")
    find_library( Cocoa_LIBRARY NAMES Cocoa )
    find_library( OpenGL_LIBRARY NAMES OpenGL )
    find_library( IOKit_LIBRARY NAMES IOKit )
    find_library( CoreFoundation_LIBRARY NAMES CoreFoundation )
    find_library( CoreVideo_LIBRARY NAMES CoreVideo )

    set( LIBRARIES 
               ${LIBRARIES} 
               ${Cocoa_LIBRARY}
               ${OpenGL_LIBRARY}
               ${IOKit_LIBRARY}
               ${CoreFoundation_LIBRARY} 
               ${CoreVideo_LIBRARY} )
endif(APPLE)

if(UNIX AND NOT APPLE)
    message("UNIX AND NOT APPLE")
    find_library( OpenGL_LIBRARY NAMES GL )

    set( LIBRARIES 
               ${LIBRARIES} 
               ${GLFW_LIBRARIES} 
               ${OpenGL_LIBRARY}
               )
endif(UNIX AND NOT APPLE)



set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}

   ${ImGui_SOURCE_DIR}/examples/opengl3_example
   #${ImGui_SOURCE_DIR}/examples/libs/gl3w
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
)

set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
)




message("INCLUDE_DIRS:${INCLUDE_DIRS}")

include_directories(${INCLUDE_DIRS})
add_definitions(${DEFINTIONS})
add_library(${name} SHARED 
imgui.cpp
examples/opengl3_example/imgui_impl_glfw_gl3.cpp
#examples/libs/gl3w/GL/gl3w.c
)

SET(hdrs 
imgui.h
imconfig.h
examples/opengl3_example/imgui_impl_glfw_gl3.h
)

target_link_libraries(${name} ${LIBRARIES})
install(TARGETS ${name}        DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

#install(FILES 
#examples/libs/gl3w/GL/gl3w.h
#examples/libs/gl3w/GL/glcorearb.h
#DESTINATION include/GL) 


add_executable(${name}Example  examples/opengl3_example/main.cpp)
target_link_libraries(${name}Example ${LIBRARIES} ${name})
install(TARGETS ${name}Example DESTINATION bin)




