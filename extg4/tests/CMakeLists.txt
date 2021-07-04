cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name ExtG4Test)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


set(TEST_SOURCES

    X4Test.cc
    X4EntityTest.cc
    X4SolidTest.cc
    X4SolidLoadTest.cc
    X4MeshTest.cc
    X4SolidExtentTest.cc
    X4SolidListTest.cc
    X4PhysicsVectorTest.cc
    X4MaterialTest.cc
    X4MaterialWaterStandaloneTest.cc
    X4MaterialWaterTest.cc
    X4MaterialTableTest.cc
    X4PhysicalVolumeTest.cc
    X4PhysicalVolume2Test.cc
    X4Transform3DTest.cc
    X4AffineTransformTest.cc
    X4ThreeVectorTest.cc
    X4CSGTest.cc
    X4PolyconeTest.cc

    X4GDMLParserTest.cc
    X4GDMLBalanceTest.cc

    X4GDMLReadStructureTest.cc
    X4GDMLReadStructure2Test.cc
    X4DumpTest.cc 
    X4GDMLReadDefineTest.cc
    G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc

    X4NameOrderTest.cc

    X4PhysicsOrderedFreeVectorTest.cc
    X4MaterialPropertyVectorTest.cc

    X4ScintillationTest.cc
)


foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    set(testname ${name}.${TGT})
    add_test(${testname} ${TGT})

    target_link_libraries(${TGT} ExtG4 )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



