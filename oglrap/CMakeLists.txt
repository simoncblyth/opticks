cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OGLRap)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLM  REQUIRED)
find_package(PLog REQUIRED) 
find_package(OpticksBoost REQUIRED) 

find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(OpticksCore REQUIRED)
find_package(GGeo REQUIRED)  
find_package(NPY REQUIRED) 
find_package(ImGui REQUIRED)


set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}/inc

   ${GLEW_INCLUDE_DIRS} 
   ${GLFW_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS} 
   ${GGeo_INCLUDE_DIRS} 
   ${PLog_INCLUDE_DIRS} 
   ${OpticksBoost_INCLUDE_DIRS} 
   ${OpticksCore_INCLUDE_DIRS}
   ${NPY_INCLUDE_DIRS}
   ${SysRap_INCLUDE_DIRS} 
   ${BoostRap_INCLUDE_DIRS} 
   ${ImGui_INCLUDE_DIRS}
) 

# generate define header for hardcoding in the library 
set(OGLRAP_INSTALL_PREFIX     "${CMAKE_INSTALL_PREFIX}")
set(OGLRAP_SHADER_DIR         "${CMAKE_INSTALL_PREFIX}/gl")
set(OGLRAP_SHADER_DYNAMIC_DIR "${CMAKE_INSTALL_PREFIX}/gl")
set(OGLRAP_SHADER_INCL_PATH   "${CMAKE_INSTALL_PREFIX}/gl")
configure_file( Config.hh.in inc/Config.hh )


set(DEFINITIONS
   ${GLEW_DEFINITIONS}
   ${GLFW_DEFINITIONS}
   ${GGeo_DEFINITIONS}
   ${OpticksCore_DEFINITIONS}
   ${PLog_DEFINITIONS}
   ${OpticksBoost_DEFINITIONS}
   ${SysRap_DEFINITIONS} 
   ${BoostRap_DEFINITIONS} 
   ${ImGui_DEFINITIONS}
   -DGUI_
)

set(LIBRARIES
   ${GLEW_LIBRARIES}
   ${GLFW_LIBRARIES}
   ${OpticksCore_LIBRARIES}
   ${GGeo_LIBRARIES}
   ${PLog_LIBRARIES}
   ${OpticksBoost_LIBRARIES}
   ${SysRap_LIBRARIES}
   ${BoostRap_LIBRARIES}
   ${NPY_LIBRARIES}
   ${ImGui_LIBRARIES}
)

if(WIN32)
   set( LIBRARIES ${LIBRARIES} opengl32.lib )
endif(WIN32)

#message("${name}.LIBRARIES : ${LIBRARIES} " )


include_directories(${INCLUDE_DIRS})
add_definitions(${DEFINITIONS})

## NB only classes needing OpenGL should be here 

set(_SOURCES

)

set(SOURCES

    OGLRAP_LOG.cc

    Texture.cc
    Device.cc
    Shdr.cc
    Prog.cc
    DynamicDefine.cc
    Colors.cc
    Demo.cc
    Touchable.cc
    StateGUI.cc
    Photons.cc
    GUI.cc

    RendererBase.cc
    Rdr.cc
    InteractorCfg.cc
    Interactor.cc
    Scene.cc
    SceneCfg.cc

    Frame.cc

    Renderer.cc
    RendererCfg.cc
    AxisApp.cc
)

SET(HEADERS

    OGLRAP_LOG.hh
    OGLRAP_API_EXPORT.hh
    OGLRAP_HEAD.hh
    OGLRAP_TAIL.hh

    Texture.hh
    Frame.hh
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

    StateGUI.hh
    Demo.hh
    Prog.hh
    Shdr.hh
    Touchable.hh
    GUI.hh
    Photons.hh
    Colors.hh
    DynamicDefine.hh

    gleq.h
    loadPPM.h
)

add_library( ${name}  SHARED ${SOURCES})
target_link_libraries( ${name} ${LIBRARIES})

install(TARGETS ${name} DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/${name})
install(DIRECTORY gl/ DESTINATION gl)

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)

