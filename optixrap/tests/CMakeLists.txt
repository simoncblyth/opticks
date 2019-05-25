cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OptiXRapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(TEST_SOURCES

    OContextCreateTest.cc
    OScintillatorLibTest.cc

    LTOOContextUploadDownloadTest.cc
    OInterpolationTest.cc

    intersect_analytic_test.cc
    Roots3And4Test.cc
)


set(PAIRED_TEST_SOURCES  # tests with paired cc and cu

    bufferTest.cc
    textureTest.cc
    boundaryTest.cc
    boundaryLookupTest.cc
    texTest.cc
    tex0Test.cc
    minimalTest.cc
    rayleighTest.cc
    writeBufferTest.cc
    writeBufferLowLevelTest.cc
    redirectLogTest.cc
    downloadTest.cc
    eventTest.cc

)


foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} OptiXRap  )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


foreach(SRC ${PAIRED_TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)

    CUDA_WRAP_SRCS( ${TGT} PTX _generated_PTX_files cu/${TGT}.cu )
    CUDA_WRAP_SRCS( ${TGT} OBJ _generated_OBJ_files ${TGT}.cc )  
    add_executable( ${TGT} ${_generated_OBJ_files} ${_generated_PTX_files} ${SRC} )
    add_test(${name}.${TGT} ${TGT})

    message(STATUS "SRC:${SRC} TGT:${TGT}  _generated_PTX_files:${_generated_PTX_files} _generated_OBJ_files:${_generated_OBJ_files} ")

    install(FILES ${_generated_PTX_files} DESTINATION installcache/PTX/tests)
    target_link_libraries(${TGT} OptiXRap  )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

#[=[
Applied the building pattern of building the OptiXRap lib 
to building all the test executables.
#]=]


#[=[
see ~/opticks/ok/tests/CMakeLists.txt for 
hints at how to add tests for the same executable
with different arguments for different shapes:
intersect_analytic_test --cu intersect_analytic_sphere_test.cu 
#]=]

