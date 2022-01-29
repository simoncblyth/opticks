cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSGTest)
project(${name} VERSION 0.1.0)

find_package(OpticksCore REQUIRED CONFIG)

set(TEST_SOURCES
    CSGNodeTest.cc
    CSGIntersectSolidTest.cc
    CSGPrimImpTest.cc
    CSGPrimSpecTest.cc
    CSGPrimTest.cc
    CSGSolidTest.cc
    CSGFoundryTest.cc
    CSGFoundry_getCenterExtent_Test.cc
    CSGFoundry_findSolidIdx_Test.cc
    CSGNameTest.cc
    CSGTargetTest.cc
    CSGTargetGlobalTest.cc
    CSGGenstepTest.cc
    CSGScanTest.cc
    CUTest.cc
    CSGLogTest.cc
    intersect_node_phicut_test.cc 
    intersect_node_box3_test.cc 
    intersect_node_cylinder_test.cc 
    CSGNodeScanTest.cc
    CSGSignedDistanceFieldTest.cc
    distance_node_test.cc

    CSGGeometryTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} Opticks::CSG Opticks::OpticksCore)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


set(TGT CSGDemoTest)
add_executable(${TGT} CSGDemoTest.cc DemoGeo.cc DemoGrid.cc)
target_link_libraries(${TGT} Opticks::CSG Opticks::OpticksCore)
install(TARGETS ${TGT} DESTINATION lib)


