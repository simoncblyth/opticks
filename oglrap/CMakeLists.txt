cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OGLRap)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(ImGui       REQUIRED MODULE)
find_package(OpticksGLEW REQUIRED MODULE)

find_package(OpticksGLFW REQUIRED MODULE)
# should come along with ImGui ?

find_package(OpticksGeo  REQUIRED CONFIG)



## NB only classes needing OpenGL should be here 

set(SOURCES

    OGLRAP_LOG.cc

    Texture.cc
    Device.cc
    Shdr.cc
    Prog.cc
    ProgLog.cc
    Colors.cc
    Demo.cc
    Touchable.cc
    StateGUI.cc
    Photons.cc
    GUI.cc

    RBuf.cc
    RBuf4.cc
    RendererBase.cc
    Rdr.cc
    InteractorCfg.cc
    Interactor.cc
    Scene.cc
    SceneCfg.cc

    G.cc
    Frame.cc
    Pix.cc

    Renderer.cc
    InstLODCull.cc
    RContext.cc
    RendererCfg.cc
    AxisApp.cc
    OpticksViz.cc
)

SET(HEADERS

    OGLRAP_LOG.hh
    OGLRAP_API_EXPORT.hh
    OGLRAP_HEAD.hh
    OGLRAP_TAIL.hh

    Texture.hh
    Frame.hh
    Pix.hh

    Interactor.hh
    InteractorCfg.hh
    Scene.hh
    SceneCfg.hh
    RendererBase.hh
    Renderer.hh
    RendererCfg.hh
    Rdr.hh
    Device.hh
    AxisApp.hh
    OpticksViz.hh

    StateGUI.hh
    Demo.hh
    Prog.hh
    G.hh
    Shdr.hh
    Touchable.hh
    GUI.hh
    Photons.hh
    Colors.hh

    gleq.h
    old_gleq.h
    GLEQ.hh
    loadPPM.h
)

add_library( ${name}  SHARED ${SOURCES})
target_link_libraries( ${name} PUBLIC 
    Opticks::OpticksGLEW
    Opticks::ImGui
    Opticks::OpticksGeo
    Opticks::OpticksGLFW
)




if(WIN32)
   target_link_libraries( ${name} PUBLIC opengl32.lib )
endif()
if(UNIX AND NOT APPLE)
   target_link_libraries( ${name} PUBLIC GL )
endif()


target_include_directories( ${name} PUBLIC 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> 
   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/inc> 
)

target_compile_definitions( ${name} PUBLIC GUI_ OPTICKS_OGLRAP ) 

install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY gl/ DESTINATION gl)

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)

add_subdirectory(tests)


