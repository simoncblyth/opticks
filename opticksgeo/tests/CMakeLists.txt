cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpticksGeoTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(TEST_SOURCES
    OpticksGeoTest.cc
    OpticksHubTest.cc
    OpticksHubGGeoTest.cc

    SensorLibTest.cc
    MockSensorAngularEfficiencyTableTest.cc
    MockSensorLibTest.cc
)

set(FORMER_TEST_SOURCES
    OpenMeshRapTest.cc
)

foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

    target_link_libraries(${TGT} OpticksGeo)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


# cmakex-
#set_property(TEST ${name}.OpticksResourceTest PROPERTY ENVIRONMENT 
#   "OPTICKS_GEOKEY=DAE_NAME_SPANNER"
#   "DAE_NAME_SPANNER=/tmp/spanner_in_works.dae"
# )

