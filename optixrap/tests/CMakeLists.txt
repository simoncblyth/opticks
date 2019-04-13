cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OptiXRapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(TEST_SOURCES
    OContextCreateTest.cc
    OScintillatorLibTest.cc
    OOTextureTest.cc
    OOMinimalTest.cc
    OOMinimalRedirectTest.cc
    OOContextTest.cc
    OOContextLowTest.cc
    OOContextUploadDownloadTest.cc
    LTOOContextUploadDownloadTest.cc

    OOboundaryTest.cc
    OOboundaryLookupTest.cc
    OOtex0Test.cc
    OOtexTest.cc

    bufferTest.cc
    OEventTest.cc
    OInterpolationTest.cc
    ORayleighTest.cc

    intersect_analytic_test.cc
    Roots3And4Test.cc
)

#[=[
see ~/opticks/ok/tests/CMakeLists.txt for 
hints at how to add tests for the same executable
with different arguments for different shapes:
intersect_analytic_test --cu intersect_analytic_sphere_test.cu 
#]=]


foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

    target_link_libraries(${TGT} OptiXRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


