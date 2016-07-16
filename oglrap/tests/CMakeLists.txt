cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OGLRapTest)
project(${name})

#file(GLOB TEST_CC_SRCS "*Test.cc")

set(FRAGILE_TEST_SOURCES
) 


set(TEST_SOURCES

    DynamicDefineTest.cc
    FrameTest.cc
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
       add_dependencies(check ${TGT})
    endif()

    target_link_libraries(${TGT} ${LIBRARIES} OGLRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



