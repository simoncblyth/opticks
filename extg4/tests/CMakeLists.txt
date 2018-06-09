cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name ExtG4Test)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


set(TEST_SOURCES
    X4SolidTest.cc
    X4PhysicsVectorTest.cc
    X4MaterialTest.cc
    X4MaterialTableTest.cc
    X4PhysicalVolumeTest.cc
    X4Transform3DTest.cc
)


foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    set(testname ${name}.${TGT})
    add_test(${testname} ${TGT})

    target_link_libraries(${TGT} ExtG4 )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



