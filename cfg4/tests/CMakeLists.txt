cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name cfg4Test)
project(${name})

#file(GLOB TEST_SOURCES "*Test.cc")

set(_TEST_SOURCES

)

set(TEST_SOURCES

    CMaterialLibTest.cc
    CTestDetectorTest.cc
    CGDMLDetectorTest.cc
    CGeometryTest.cc
    CG4Test.cc
    G4MaterialTest.cc
    G4StringTest.cc
    G4BoxTest.cc
    CCollectorTest.cc
    CInterpolationTest.cc
    OpRayleighTest.cc 
    GROUPVELTest.cc 

)


foreach(SRC ${TEST_SOURCES})

    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} ${LIBRARIES} cfg4)
    install(TARGETS ${TGT} DESTINATION lib)

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

endforeach()

