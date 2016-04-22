cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name Opticks)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(Boost REQUIRED)
find_package(GLM REQUIRED)

find_package(Bregex REQUIRED)
find_package(Cfg REQUIRED)
find_package(NPY REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS}  

   ${Bregex_INCLUDE_DIRS}  
   ${Cfg_INCLUDE_DIRS}  
   ${NPY_INCLUDE_DIRS}  
)

add_definitions( 
   ${Boost_DEFINITIONS} 
   ${GLM_DEFINITIONS} 
   ${NPY_DEFINITIONS} 
   ${Cfg_DEFINITIONS} 
   ${Bregex_DEFINITIONS} 
    -DVERBOSE 
)

set(LIBRARIES
      ${Boost_LIBRARIES}
      ${GLM_LIBRARIES}

      NPY
      Cfg
      Bregex
)

add_library( ${name}  SHARED
    OpticksResource.cc
    Opticks.cc
    OpticksCfg.cc

    Composition.cc
    CompositionCfg.cc
    View.cc
    ViewCfg.cc
    Camera.cc
    CameraCfg.cc
    Trackball.cc
    TrackballCfg.cc
    Clipper.cc
    ClipperCfg.cc
    InterpolatedView.cc
    OrbitalView.cc
    TrackView.cc

    Animator.cc
    Bookmarks.cc
    Light.cc
)

SET(hdrs 
    OpticksResource.hh
    Opticks.hh
    OpticksCfg.hh

    Composition.hh
    CompositionCfg.hh
    View.hh
    ViewCfg.hh
    Camera.hh
    CameraCfg.hh
    Trackball.hh
    TrackballCfg.hh
    Clipper.hh
    ClipperCfg.hh
    InterpolatedView.hh
    OrbitalView.hh
    TrackView.hh

    Animator.hh
    Bookmarks.hh
    Light.hh

    OpticksPhoton.h
    OpticksFlags.h
)

target_link_libraries( ${name} ${LIBRARIES} )
install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs} DESTINATION include/${name})

add_subdirectory(tests)

