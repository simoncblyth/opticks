cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
# this file is copied into place by imgui-cmake-ize
set(name ImGui)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)

set(LIBRARIES
   ${GLEW_LIBRARIES} 
   ${GLFW_LIBRARIES}
)

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


if(WIN32)
   message("WIN32....")
endif(WIN32)


if(WIN64)
   message("WIN64....")
endif(WIN64)


if(MSYS)
   message("MSYS....")
endif(MSYS)


if(MINGW)
   message("MINGW....")
endif(MINGW)







set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${ImGui_SOURCE_DIR}/examples/opengl3_example
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
)

set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
)

set(SOURCES
   imgui.cpp
   imgui_draw.cpp
   imgui_demo.cpp
   examples/opengl3_example/imgui_impl_glfw_gl3.cpp
)

SET(HEADERS
   imgui.h
   imconfig.h
   examples/opengl3_example/imgui_impl_glfw_gl3.h
)

include_directories(${INCLUDE_DIRS})
add_definitions(${DEFINTIONS})
add_library(${name} SHARED ${SOURCES})
target_link_libraries(${name} ${LIBRARIES})

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include)


## getting the example to build entails converting from gl3w to glew 
## thats too much hassle whilst operating in external code
## TODO: make my own example in OGLRap/tests 

if(WITH_IMGUI_EXAMPLE)
   add_executable(${name}Example  examples/opengl3_example/main.cpp)
   target_link_libraries(${name}Example ${LIBRARIES} ${name})
   install(TARGETS ${name}Example DESTINATION bin)
endif(WITH_IMGUI_EXAMPLE)

