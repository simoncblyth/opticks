cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpticksCore)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(OKConf   REQUIRED CONFIG)
find_package(NPY      REQUIRED CONFIG)


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
    OpticksAna.cc
    OpticksDbg.cc
    OpticksDomain.cc
    OpticksEventInstrument.cc
    OpticksEventSpec.cc
    OpticksEventAna.cc
    OpticksEventStat.cc
    OpticksEventCompare.cc
    OpticksEventDump.cc
    OpticksEvent.cc
    OpticksRun.cc
    OpticksEntry.cc
    OpticksBufferSpec.cc
    OpticksBufferControl.cc
    OpticksActionControl.cc
    Indexer.cc

    OpticksFlags.cc
    OpticksAttrSeq.cc
    Opticks.cc
    OpticksMode.cc
    OpticksCfg.cc
    OpticksResource.cc
    OpticksProfile.cc

    CameraCfg.cc
    ViewCfg.cc
    TrackballCfg.cc
    ClipperCfg.cc
    CompositionCfg.cc

    OpticksApp.cc
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
    OpticksAna.hh
    OpticksDbg.hh
    OpticksDomain.hh
    OpticksEventSpec.hh
    OpticksEvent.hh
    OpticksEventAna.hh
    OpticksRun.hh
    OpticksEntry.hh
    OpticksBufferControl.hh
    OpticksActionControl.hh
    OpticksBufferSpec.hh
    Indexer.hh

    OpticksSwitches.h
    OpticksFlags.hh
    OpticksAttrSeq.hh
    Opticks.hh
    OpticksMode.hh
    OpticksCfg.hh
    OpticksResource.hh
    OpticksProfile.hh

    CameraCfg.hh
    ViewCfg.hh
    TrackballCfg.hh
    ClipperCfg.hh
    CompositionCfg.hh

    OpticksApp.hh
    Bookmarks.hh
    Composition.hh

    OpticksPhoton.h
    OpticksQuadrant.h
)


add_library( ${name}  SHARED ${SOURCES})
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )

target_link_libraries( ${name} 
  PUBLIC 
     Opticks::NPY
     Opticks::OKConf
)


bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)

