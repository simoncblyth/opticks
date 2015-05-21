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

if(APPLE)
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


include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${ImGui_SOURCE_DIR}/examples/opengl3_example
   ${ImGui_SOURCE_DIR}/examples/libs/gl3w
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 

)

add_definitions(
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
)


add_library(${name} SHARED 
imgui.cpp
examples/opengl3_example/imgui_impl_glfw_gl3.cpp
examples/libs/gl3w/GL/gl3w.c
)

SET(hdrs 
imgui.h
imconfig.h
examples/opengl3_example/imgui_impl_glfw_gl3.h
)

target_link_libraries(${name} ${LIBRARIES})
install(TARGETS ${name}        DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

install(FILES 
examples/libs/gl3w/GL/gl3w.h
examples/libs/gl3w/GL/glcorearb.h
DESTINATION include/GL) 


add_executable(${name}Example  examples/opengl3_example/main.cpp)
target_link_libraries(${name}Example ${LIBRARIES} ${name})
install(TARGETS ${name}Example DESTINATION bin)




