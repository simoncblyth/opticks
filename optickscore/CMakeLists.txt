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

    OpticksPhoton.h
    OpticksQuadrant.h
)


add_library( ${name}  SHARED ${SOURCES} ${HEADERS} )
target_include_directories( ${name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
target_compile_definitions( ${name} PUBLIC OPTICKS_OKCORE )
target_link_libraries( ${name} 
  PUBLIC 
     Opticks::NPY
     Opticks::OKConf
)



set(HDR   ${CMAKE_CURRENT_SOURCE_DIR}/OpticksPhoton.h)
set(PYINI ${CMAKE_CURRENT_BINARY_DIR}/OpticksPhoton_Enum.ini)
add_custom_command(
   OUTPUT 
         ${PYINI}
   COMMAND 
          enum.py ${HDR} --quiet --inipath ${PYINI}
   DEPENDS 
         ${HDR}
)
add_custom_target(PythonINI ALL DEPENDS ${PYINI} )



set(CCFL ${CMAKE_CURRENT_SOURCE_DIR}/OpticksFlags.cc)
set(PYJS ${CMAKE_CURRENT_BINARY_DIR}/OpticksFlags_Abbrev.json)
add_custom_command(
   OUTPUT 
         ${PYJS}
   COMMAND 
          OpticksFlags.py ${CCFL} --quiet --jsonpath ${PYJS}
   DEPENDS 
         ${CCFL}
)
add_custom_target(PythonJSON ALL DEPENDS ${PYJS} )

list(APPEND HEADERS 
  ${PYINI} 
  ${PYJS}
) 


bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)


#[=[


Linker Visibility Warnings (FIXED?)
--------------------------------------

::

    [ 50%] Linking CXX shared library libOpticksCore.dylib
    ld: warning: direct access in function
                        'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_function_call>>::rethrow() const' 
    from file 
           'CMakeFiles/OpticksCore.dir/Clipper.cc.o' 
    to global weak symbol 
           'typeinfo for boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_function_call>>' 
    from file 
           'CMakeFiles/OpticksCore.dir/OpticksCfg.cc.o' 
    means the weak symbol cannot be overridden at runtime. 
    This was likely caused by different translation units being compiled with different visibility settings.  [ 50%]


Were still getting some of the above warning following addition of -fvisibility-inlines-hidden 
Observed that only symbols three classes implicated::

   Clipper
   Trackball 
   OpticksAttrSeq 


All three were using boost::split, moving to BStr::split avoids the warnings.


#]=]

