cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name G4CXTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(TEST_SOURCES
   G4CXRenderTest.cc
   G4CXSimulateTest.cc
)


foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} G4CX)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


