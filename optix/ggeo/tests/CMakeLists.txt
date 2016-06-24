cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name GGeoTest)
project(${name})

#file(GLOB TEST_SOURCES "*Test.cc")
set(_TEST_SOURCES

    GConstantTest.cc
    GDomainTest.cc
    GAryTest.cc

    GBufferTest.cc
    GVectorTest.cc

    GPropertyMapBaseTest.cc
    GPropertyBaseTest.cc

    GItemListTest.cc
    GMaterialLibTest.cc

    GSkinSurfaceTest.cc
    GBorderSurfaceTest.cc
    GScintillatorLibTest.cc
    GSourceTest.cc
    GSourceLibTest.cc
    GBndLibTest.cc

    GMeshTest.cc
    GNodeTest.cc
    GSolidTest.cc
    GCIETest.cc
    GArrayTest.cc
    GTransformsTest.cc
    GIdsTest.cc
    GItemIndexTest.cc
    GTreePresentTest.cc
    GCSGTest.cc
    GPartsTest.cc
    GPmtTest.cc

    BoundariesNPYTest.cc
    GAttrSeqTest.cc
    GBBoxMeshTest.cc

    GColorsTest.cc
    GFlagsTest.cc
    GGeoLibTest.cc

    GGeoTest.cc
    GGeoTestTest.cc
    GMakerTest.cc

    GMergedMeshTest.cc
    GMatrixTest.cc
    GMaterialTest.cc

    GPropertyDebugTest.cc
    GPropertyMapTest.cc

    GPropertyTest.cc
    GSurfaceLibTest.cc
    GTestBoxTest.cc

    LookupTest.cc
    RecordsNPYTest.cc
    TorchStepNPYTest.cc

)

set(TEST_SOURCES
    GBufferTest.cc
)


foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})
    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})
    target_link_libraries(${TGT} ${LIBRARIES} GGeo)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



