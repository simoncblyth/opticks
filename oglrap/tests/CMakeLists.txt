cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OGLRapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(FRAGILE_TEST_SOURCES
    FrameTest.cc
) 

set(TEST_SOURCES

    ModelToWorldTest.cc
    ProgTest.cc

    InteractorKeys.cc
    AxisAppCheck.cc
    SceneCheck.cc

    gleq_check.c
)


foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    if (${TGT} MATCHES ".*Test$")  
       set(testname ${name}.${TGT})
       add_test(${testname} ${TGT})
    endif()

    target_link_libraries(${TGT} OGLRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



