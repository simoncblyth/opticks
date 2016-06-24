cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OpticksCore)
project(${name})

include(EnvBuildOptions)
include(EnvCompilationFlags)

find_package(OpticksBoost REQUIRED)
find_package(GLM REQUIRED)

find_package(PLog REQUIRED)
find_package(SysRap REQUIRED)
find_package(BoostRap REQUIRED)
find_package(NPY REQUIRED)

include_directories(
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${CMAKE_CURRENT_BINARY_DIR}/inc

   ${PLog_INCLUDE_DIRS} 
   ${OpticksBoost_INCLUDE_DIRS} 
   ${GLM_INCLUDE_DIRS}  

   ${SysRap_INCLUDE_DIRS}  
   ${BoostRap_INCLUDE_DIRS}  
   ${NPY_INCLUDE_DIRS}  
)

add_definitions( 
   ${OpticksBoost_DEFINITIONS} 
   ${PLog_DEFINITIONS} 
   ${GLM_DEFINITIONS} 
   ${NPY_DEFINITIONS} 
   ${SysRap_DEFINITIONS} 
   ${BoostRap_DEFINITIONS} 
    -DVERBOSE 
)

set(LIBRARIES
   ${OpticksBoost_LIBRARIES}
   ${PLog_LIBRARIES} 
   ${GLM_LIBRARIES}

   ${NPY_LIBRARIES}
   ${SysRap_LIBRARIES}
   ${BoostRap_LIBRARIES}
)


# generate define header for hardcoding in the library 
set(OPTICKS_INSTALL_PREFIX     "${CMAKE_INSTALL_PREFIX}")
configure_file( OpticksCMakeConfig.hh.in inc/OpticksCMakeConfig.hh )

set(SOURCES
    OKCORE_LOG.cc

    Animator.cc
    View.cc
    Clipper.cc
    Camera.cc
    Light.cc
    Trackball.cc
    InterpolatedView.cc
    OrbitalView.cc
    TrackView.cc

    OpticksConst.cc
    OpticksQuery.cc
    OpticksColors.cc
    Sparse.cc
    OpticksEvent.cc
    Indexer.cc

    OpticksFlags.cc
    OpticksAttrSeq.cc
    Opticks.cc
    OpticksCfg.cc
    OpticksResource.cc

    CameraCfg.cc
    ViewCfg.cc
    TrackballCfg.cc
    ClipperCfg.cc
    CompositionCfg.cc

    OpticksEngine.cc
    Bookmarks.cc
    Composition.cc

    Demo.cc
    DemoCfg.cc
)


set(_SOURCES

) 


set(HEADERS

    OKCORE_LOG.hh
    OKCORE_API_EXPORT.hh
    OKCORE_HEAD.hh
    OKCORE_TAIL.hh

    Animator.hh
    View.hh
    Clipper.hh
    Camera.hh
    Light.hh
    Trackball.hh
    InterpolatedView.hh
    OrbitalView.hh
    TrackView.hh

    OpticksConst.hh
    OpticksQuery.hh
    OpticksColors.hh
    Sparse.hh
    OpticksEvent.hh
    Indexer.hh

    OpticksFlags.hh
    OpticksAttrSeq.hh
    Opticks.hh
    OpticksCfg.hh
    OpticksResource.hh

    CameraCfg.hh
    ViewCfg.hh
    TrackballCfg.hh
    ClipperCfg.hh
    CompositionCfg.hh

    OpticksEngine.hh
    Bookmarks.hh
    Composition.hh

    OpticksPhoton.h
    OpticksFlags.h
)


add_library( ${name}  SHARED ${SOURCES}  )
target_link_libraries( ${name} ${LIBRARIES} )
install(TARGETS ${name}  DESTINATION lib)
install(FILES ${hdrs} DESTINATION include/${name})

#add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tests)

