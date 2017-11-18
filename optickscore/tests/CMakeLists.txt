cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OpticksCoreTest)
project(${name})

#file(GLOB TEST_SOURCES "*Test.cc")

set(TEST_SOURCES

    OpticksQueryTest.cc
    IndexerTest.cc
    CameraTest.cc
    AnimatorTest.cc
    TrackballTest.cc
    OpticksFlagsTest.cc
    OpticksPhotonTest.cc
    OpticksCfgTest.cc
    OpticksTest.cc
    OpticksResourceTest.cc
    OpticksBufferControlTest.cc
    OpticksBufferSpecTest.cc
    OpticksActionControlTest.cc

    OpticksIDPATH.cc
    OpticksPrepareInstallCache.cc
    OpticksEventSpecTest.cc
    OpticksAnaTest.cc
    OpticksDbgTest.cc

    CompositionTest.cc
    ViewTest.cc

    EvtLoadTest.cc
    OpticksEventAnaTest.cc
    OpticksEventCompareTest.cc
    OpticksEventDumpTest.cc

)


set(TEST_SOURCES_NOK 

    OpticksEventTest.cc
    OpticksAppTest.cc

    NGunConfigTest.cc

    BookmarksTest.cc
    InterpolatedViewTest.cc
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_SOURCE} NAME_WE)
    add_executable(${TGT} ${TEST_SOURCE})

    if (${TGT} MATCHES ".*Test$")  
       set(testname ${name}.${TGT})
       add_test(${testname} ${TGT})
       add_dependencies(check ${TGT})
    else()
       #message(" skip non-test ${TGT} ")
    endif()

    target_link_libraries(${TGT} ${LIBRARIES} OpticksCore)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


# cmakex-
#set_property(TEST ${name}.OpticksResourceTest PROPERTY ENVIRONMENT 
#   "OPTICKS_GEOKEY=DAE_NAME_SPANNER"
#   "DAE_NAME_SPANNER=/tmp/spanner_in_works.dae"
# )

