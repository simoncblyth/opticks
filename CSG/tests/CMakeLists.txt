cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSGTest)
project(${name} VERSION 0.1.0)


set(TEST_SOURCES
    CSGNodeTest.cc
    CSGPrimTest.cc
    CSGSolidTest.cc
    CSGFoundryTest.cc
    CSGFoundryLoadTest.cc
    CSGNameTest.cc
    CSGTargetTest.cc
    CSGScanTest.cc
    TranTest.cc
    CUTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} CSG)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


set(TGT CSGDemoTest)
add_executable(${TGT} CSGDemoTest.cc DemoGeo.cc DemoGrid.cc)
target_link_libraries(${TGT} CSG)
install(TARGETS ${TGT} DESTINATION lib)


