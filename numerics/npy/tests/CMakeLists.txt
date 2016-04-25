cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPYTest)
project(${name})

file(GLOB TEST_CC_SRCS "*Test.cc")

foreach(TEST_CC_SRC ${TEST_CC_SRCS})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    # see cmakex-
    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} 
               ${LIBRARIES} 
               NPY
    )
    install(TARGETS ${TGT} DESTINATION bin)
endforeach()

