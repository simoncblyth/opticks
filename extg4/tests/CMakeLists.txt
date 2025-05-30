cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name ExtG4Test)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(PMTSim_standalone CONFIG)
message( STATUS " PMTSim_standalone_FOUND : ${PMTSim_standalone_FOUND} " )


set(PMTSIM_TEST_SOURCES
    X4IntersectSolidTest.cc 
    X4SimtraceTest.cc 
    X4IntersectVolumeTest.cc 
    X4MeshTest.cc
    X4VolumeMakerTest.cc 
)

# hmm what about executables than can use PMTSim but are also useful without ?

set(TEST_SOURCES

    X4SolidMakerTest.cc
    X4SolidMultiUnionTest.cc 

    X4Test.cc
    X4EntityTest.cc
    X4SolidTest.cc
    X4SolidLoadTest.cc
    X4SolidTreeTest.cc

    X4MeshTest0.cc

    X4SolidExtentTest.cc
    X4SolidListTest.cc
    X4PhysicsVectorTest.cc
    X4MaterialTest.cc
    X4MaterialWaterStandaloneTest.cc
    X4MaterialWaterTest.cc
    X4MaterialTableTest.cc
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

    X4ArrayTest.cc
    X4MaterialPropertyVectorTest.cc
    X4MaterialPropertiesTableTest.cc

    X4ScintillationTest.cc
    PhysicsFreeVectorTest.cc 
    G4MaterialPropertyVectorTest.cc 
    X4PhysicalConstantsTest.cc


    CLHEPVersionInteger.cc

    convertMultiUnionTest.cc
)

set(FORMER_TEST_SOURCES

    X4PhysicalVolumeTest.cc   ## X4PhysicalVolume::Convert no longer used NOW USING X4Geo::Translate
    X4PhysicalVolume2Test.cc
    X4SurfaceTest.cc

)


foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    set(testname ${name}.${TGT})
    add_test(${testname} ${TGT})

    #target_compile_definitions( ${TGT} PUBLIC G4MULTITHREADED G4USE_STD11 )
    target_link_libraries(${TGT} PUBLIC ExtG4 )

    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



foreach(TEST_CC_SRC ${PMTSIM_TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    set(testname ${name}.${TGT})
    add_test(${testname} ${TGT})

    target_link_libraries(${TGT} PUBLIC ExtG4 )

    if(PMTSim_standalone_FOUND)
       target_link_libraries( ${TGT} PUBLIC Opticks::PMTSim_standalone )
       target_compile_definitions( ${TGT} PUBLIC WITH_PMTSIM )
    endif()

    install(TARGETS ${TGT} DESTINATION lib)
endforeach()







