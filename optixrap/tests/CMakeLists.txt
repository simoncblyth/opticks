cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OptiXRapTest)
project(${name})

#file(GLOB TEST_CC_SRCS "*Test.cc")
set(TEST_SOURCES
    OPropertyLibTest.cc
    OScintillatorLibTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} OptiXRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


