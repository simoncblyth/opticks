cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OGLRap)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)

###
### CAUTION: this is still not using C++11 but  
###          most other packages are 
###          there is potential for interface difficulties
###          (this might be the cause of std::string in interface problems
###          that had before : dont recall exactly where)   
###

########### using env/cmake/Modules/*.cmake #####

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLM  REQUIRED)
find_package(Boost REQUIRED) 
find_package(Cfg REQUIRED) 
find_package(Opticks REQUIRED)

find_package(GGeo REQUIRED)  
find_package(PPM REQUIRED) 
find_package(NPY REQUIRED) 
find_package(Bregex REQUIRED)

find_package(ImGui REQUIRED)


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}/inc
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${GGeo_INCLUDE_DIRS} 
   ${Boost_INCLUDE_DIRS} 
   ${Cfg_INCLUDE_DIRS}
   ${Opticks_INCLUDE_DIRS}
   ${PPM_INCLUDE_DIRS}
   ${NPY_INCLUDE_DIRS}
   ${Bregex_INCLUDE_DIRS} 
   ${ImGui_INCLUDE_DIRS}
) 


# generate define header for hardcoding in the library 
set(OGLRAP_INSTALL_PREFIX     "${CMAKE_INSTALL_PREFIX}")
set(OGLRAP_SHADER_DIR         "${CMAKE_INSTALL_PREFIX}/gl")
set(OGLRAP_SHADER_DYNAMIC_DIR "${CMAKE_INSTALL_PREFIX}/gl")
set(OGLRAP_SHADER_INCL_PATH   "${CMAKE_INSTALL_PREFIX}/gl")
configure_file( Config.hh.in inc/Config.hh )


message( "CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} ")

set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${GGeo_DEFINITIONS}
   ${Opticks_DEFINITIONS}
   ${Boost_DEFINITIONS}
   ${Bregex_DEFINITIONS} 
   ${ImGui_DEFINITIONS}
   -DGUI_
)

set(LIBRARIES
   ${GLEW_LIBRARIES}
   ${GLFW_LIBRARIES}
   ${Opticks_LIBRARIES}
   ${GGeo_LIBRARIES}
   ${Boost_LIBRARIES}
   ${Bregex_LIBRARIES}
   ${Cfg_LIBRARIES}
   ${NPY_LIBRARIES}
   
   ${ImGui_LIBRARIES}
)


include_directories(${INCLUDE_DIRS})
add_definitions(${DEFINITIONS})

# TODO: 
#    start migrating classes that do not depend on OpenGL 
#    elsewhere ... eg into opticks
#

add_library( ${name}  SHARED
                Composition.cc
                CompositionCfg.cc
                Texture.cc
                Frame.cc
                Interactor.cc
                InteractorCfg.cc
                Scene.cc
                SceneCfg.cc
                RendererBase.cc
                Renderer.cc
                RendererCfg.cc
                Device.cc
                Rdr.cc
                Trackball.cc
                TrackballCfg.cc
                Camera.cc
                CameraCfg.cc


                Light.cc
                Clipper.cc
                ClipperCfg.cc
                Bookmarks.cc
                StateGUI.cc
                Demo.cc
                Prog.cc
                Shdr.cc
                Touchable.cc
                GUI.cc
                Photons.cc
                Colors.cc
                DynamicDefine.cc
)

SET(hdrs 
        Composition.hh
        CompositionCfg.hh
        Texture.hh
        Frame.hh
        Interactor.hh
        InteractorCfg.hh
        Scene.hh
        SceneCfg.hh
        RendererBase.hh
        Device.hh
        Rdr.hh
        Renderer.hh
        RendererCfg.hh
        Trackball.hh
        TrackballCfg.hh
        Camera.hh
        CameraCfg.hh


        Light.hh
        Clipper.hh
        ClipperCfg.hh
        Bookmarks.hh
        StateGUI.hh
        Demo.hh
        gleq.h
        Prog.hh
        Shdr.hh
        Touchable.hh
        GUI.hh
        Photons.hh
        Colors.hh
        DynamicDefine.hh
)


target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)
install(DIRECTORY gl/ DESTINATION gl)


add_subdirectory(tests)




