cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpticksCore)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

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
    FlightPath.cc
    Snap.cc
    OrbitalView.cc
    TrackView.cc
    Ctrl.cc

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
    OpticksGenstep.cc
    Indexer.cc

    OpticksPhotonFlags.cc
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
    ContentStyle.cc
    RenderStyle.cc
    GlobalStyle.cc

    Demo.cc
    DemoCfg.cc

    IntersectSDF.cc
    OpticksIdentity.cc
    OpticksShape.cc

    SphereOfTransforms.cc

    SensorLib.cc
    MockSensorLib.cc
    MockSensorAngularEfficiencyTable.cc

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
    FlightPath.hh
    Snap.hh
    OrbitalView.hh
    TrackView.hh
    Ctrl.hh

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
    OpticksGenstep.hh
    Indexer.hh

    OpticksSwitches.h
    OpticksPhotonFlags.hh
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
    ContentStyle.hh
    RenderStyle.hh
    GlobalStyle.hh

    OpticksQuadrant.h

    IntersectSDF.hh
    OpticksIdentity.hh
    OpticksShape.hh

    SphereOfTransforms.hh

    SensorLib.hh
    MockSensorLib.hh
    MockSensorAngularEfficiencyTable.hh 
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
target_compile_definitions( ${name} PUBLIC OPTICKS_OKCORE )
target_link_libraries( ${name} 
  PUBLIC 
     Opticks::NPY
     Opticks::OKConf
)

# OpticksPhoton.h and _Enum.ini generation moved to sysrap 
# OpticksGenstep.h and _Enum_ini generation moved to sysrap
# generation of OpticksPhoton_Abbrev.json moved to sysrap

bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)


