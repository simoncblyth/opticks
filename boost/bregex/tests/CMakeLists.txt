cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BregexTest)
project(${name})

file(GLOB TEST_CC_SRCS "*Test.cc")

foreach(TEST_CC_SRC ${TEST_CC_SRCS})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})
    target_link_libraries(${TGT} ${LIBRARIES} Bregex)
    install(TARGETS ${TGT} DESTINATION bin)
endforeach()


