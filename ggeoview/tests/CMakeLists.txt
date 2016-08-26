cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name GGeoViewTest)
project(${name})

#file(GLOB TEST_CC_SRCS "*Test.cc")


set(_TESTS
)

set(TESTS

    flagsTest.cc
    OpticksMgrTest.cc
    OTracerTest.cc
    LogTest.cc
    OpEngineTest.cc
)


foreach(SRC ${TESTS})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
  
    #add_test(NAME ${name}.${TGT} COMMAND OpticksCTestRunner.sh --config $<CONFIGURATION> --exe $<TARGET_FILE:${TGT}> --remote-args --compute)
    # cmake version 3.4.1 doesnt set the appropriate exe path but cmake version 3.5.2 does  
    # (avoid the complication by moving the smarts into the executable)

    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} GGeoView)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

