cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSGTest)
project(${name} VERSION 0.1.0)

set(INSTALLED_SCRIPTS
    CSGTestRunner.sh
)
install(PROGRAMS ${INSTALLED_SCRIPTS} DESTINATION bin)

set(_TEST_SOURCES

)

set(TEST_SOURCES

    CSGNodeTest.cc
    CSGNodeImpTest.cc
    CSGIntersectSolidTest.cc

    CSGPrimImpTest.cc
    CSGPrimSpecTest.cc
    CSGPrimTest.cc
    CSGSolidTest.cc
    CSGFoundryTest.cc

    CSGFoundry_addPrimNodes_Test.cc
    CSGFoundry_getCenterExtent_Test.cc
    CSGFoundry_findSolidIdx_Test.cc

    CSGFoundry_CreateFromSimTest.cc
    CSGFoundry_IntersectPrimTest.cc

    CSGNameTest.cc
    CSGTargetTest.cc
    CSGTargetGlobalTest.cc
    CSGFoundry_MakeCenterExtentGensteps_Test.cc
    CSGFoundry_getFrame_Test.cc
    CSGFoundry_getFrameE_Test.cc
    CSGFoundry_getMeshName_Test.cc
    CSGFoundry_SGeo_SEvt_Test.cc

    CSGFoundry_ResolveCFBase_Test.cc
    CSGFoundryLoadTest.cc

    CSGScanTest.cc
    CSGLogTest.cc
    CSGMakerTest.cc
    CSGQueryTest.cc

    CSGSimtraceTest.cc
    CSGSimtraceRerunTest.cc
    CSGSimtraceSampleTest.cc

    CSGCopyTest.cc

    intersect_leaf_phicut_test.cc
    intersect_leaf_thetacut_test.cc
    intersect_leaf_box3_test.cc
    intersect_leaf_cylinder_test.cc
    intersect_leaf_cylinder_vs_oldcylinder_test.cc
    CSGIntersectComparisonTest.cc

    distance_leaf_slab_test.cc

    CSGNodeScanTest.cc
    CSGSignedDistanceFieldTest.cc

    CSGGeometryTest.cc
    CSGClassifyTest.cc
)

set(CUDA_TEST_SOURCES

    CUTest.cc
)


set( DEFERRED_TEST_SOURCES
    CSGGeometryFromGeocacheTest.cc
)

find_program(BASH_EXECUTABLE NAMES bash REQUIRED)
message(STATUS "BASH_EXECUTABLE : ${BASH_EXECUTABLE}")



foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} Opticks::CSG)
    install(TARGETS ${TGT} DESTINATION lib)

    add_test(
       NAME ${name}.${TGT}
       COMMAND bash CSGTestRunner.sh ${TGT}
    )
endforeach()


if(BUILD_WITH_CUDA)
foreach(SRC ${CUDA_TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} Opticks::CSG CUDA::cudart)
    install(TARGETS ${TGT} DESTINATION lib)

    add_test(
       NAME ${name}.${TGT}
       COMMAND bash CSGTestRunner.sh ${TGT}
    )
endforeach()
endif()

if(BUILD_WITH_CUDA) # TMP
set(TGT CSGDemoTest)
add_executable(${TGT} CSGDemoTest.cc DemoGeo.cc DemoGrid.cc)
target_link_libraries(${TGT} Opticks::CSG)     # CUDA::cudart
install(TARGETS ${TGT} DESTINATION lib)
endif()
