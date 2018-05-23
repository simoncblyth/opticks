cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OGLRapTest)
project(${name} VERSION 0.1.0)

set(FRAGILE_TEST_SOURCES
    FrameTest.cc
) 

set(TEST_SOURCES

    ModelToWorldTest.cc
    ProgTest.cc

    InteractorKeys.cc
    AxisAppCheck.cc
    SceneCheck.cc
)


foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    if (${TGT} MATCHES ".*Test$")  
       set(testname ${name}.${TGT})
       add_test(${testname} ${TGT})
       #add_dependencies(check ${TGT})
    endif()

    target_link_libraries(${TGT} OGLRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



