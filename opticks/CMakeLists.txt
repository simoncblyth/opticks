cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name Opticks)
project(${name})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} 
                      "$ENV{ENV_HOME}/cmake/Modules"
          ) 

include(EnvBuildOptions)


find_package(Boost REQUIRED)
find_package(GLM REQUIRED)
find_package(NPY REQUIRED)
find_package(Cfg REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${Boost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS}  
   ${NPY_INCLUDE_DIRS}  
   ${Cfg_INCLUDE_DIRS}  
)

add_definitions( 
   ${Boost_DEFINITIONS} 
   ${GLM_DEFINITIONS} 
   ${NPY_DEFINITIONS} 
   ${Cfg_DEFINITIONS} 
    -DVERBOSE 
)

set(LIBRARIES
      ${Boost_LIBRARIES}
      ${Cfg_LIBRARIES}
      ${GLM_LIBRARIES}
      ${NPY_LIBRARIES}
)


add_library( ${name}  SHARED
    OpticksResource.cc
    Opticks.cc
    OpticksCfg.cc
    View.cc
    ViewCfg.cc
    Camera.cc
    CameraCfg.cc

    InterpolatedView.cc
    OrbitalView.cc
    TrackView.cc
    Animator.cc
)


target_link_libraries( ${name} ${LIBRARIES} )

SET(hdrs 
    OpticksResource.hh
    Opticks.hh
    OpticksCfg.hh

    View.hh
    ViewCfg.hh
    Camera.hh
    CameraCfg.hh

    OpticksPhoton.h
    OpticksFlags.h

    InterpolatedView.hh
    OrbitalView.hh
    TrackView.hh
    Animator.hh

)

install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs}    DESTINATION include)

add_subdirectory(tests)

