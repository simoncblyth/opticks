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
find_package(GGeo REQUIRED)  # bit reluctant on this 
find_package(AssimpWrap REQUIRED)  # more reluctant still, the solution is to be able to persist/load GGeo to/from file 
find_package(Boost REQUIRED) 
find_package(Cfg REQUIRED) 
find_package(PPM REQUIRED) 
find_package(NPY REQUIRED) 

find_package(ImGui REQUIRED)


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${GGeo_INCLUDE_DIRS} 
   ${AssimpWrap_INCLUDE_DIRS} 
   ${Boost_INCLUDE_DIRS} 
   ${Cfg_INCLUDE_DIRS}
   ${PPM_INCLUDE_DIRS}
   ${NPY_INCLUDE_DIRS}
   ${ImGui_INCLUDE_DIRS}
) 

set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${GGeo_DEFINITIONS}
   ${AssimpWrap_DEFINITIONS}
   ${Boost_DEFINITIONS}
   ${ImGui_DEFINITIONS}
   -DGUI_
)

set(LIBRARIES
   ${GLEW_LIBRARIES}
   ${GLFW_LIBRARIES}
   ${GGeo_LIBRARIES}
   ${AssimpWrap_LIBRARIES}
   ${Boost_LIBRARIES}
   ${Cfg_LIBRARIES}
   ${NPY_LIBRARIES}
   ${ImGui_LIBRARIES}
)


include_directories(${INCLUDE_DIRS})
add_definitions(${DEFINITIONS})

add_library( ${name}  SHARED
                Geometry.cc
                Composition.cc
                CompositionCfg.cc
                Texture.cc
                Frame.cc
                FrameCfg.cc
                Interactor.cc
                InteractorCfg.cc
                Scene.cc
                SceneCfg.cc
                RendererBase.cc
                Renderer.cc
                RendererCfg.cc
                Rdr.cc
                Trackball.cc
                TrackballCfg.cc
                Camera.cc
                CameraCfg.cc
                View.cc
                ViewCfg.cc
                Clipper.cc
                ClipperCfg.cc
                Bookmarks.cc
                Demo.cc
                Prog.cc
                Shdr.cc
                Touchable.cc
                GUI.cc
                Photons.cc
)

SET(hdrs 
        Geometry.hh
        Composition.hh
        CompositionCfg.hh
        Texture.hh
        Frame.hh
        FrameCfg.hh
        Interactor.hh
        InteractorCfg.hh
        Scene.hh
        SceneCfg.hh
        RendererBase.hh
        Rdr.hh
        Renderer.hh
        RendererCfg.hh
        Trackball.hh
        TrackballCfg.hh
        Camera.hh
        CameraCfg.hh
        View.hh
        ViewCfg.hh
        Clipper.hh
        ClipperCfg.hh
        Bookmarks.hh
        Demo.hh
        gleq.h
        Prog.hh
        Shdr.hh
        Touchable.hh
        Configurable.hh
        GUI.hh
        Photons.hh
)


target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION lib)
install(FILES ${hdrs} DESTINATION include)

add_subdirectory(tests)




