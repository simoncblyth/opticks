cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name ThrustHelloTest)
project(${name})

file(GLOB TEST_CU_SRCS "*Test.cu")

foreach(TEST_CU_SRC ${TEST_CU_SRCS})
    get_filename_component(TGT ${TEST_CU_SRC} NAME_WE)

    cuda_add_executable(${TGT} ${TEST_CU_SRC})
    #set_target_properties(${TGT} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${TGT} ${LIBRARIES} )

    install(TARGETS ${TGT} DESTINATION bin)
endforeach()


file(GLOB TEST_CC_SRCS "*Test.cc")

foreach(TEST_CC_SRC ${TEST_CC_SRCS})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)

    add_executable(${TGT} ${TEST_CC_SRC})
    #set_target_properties(${TGT} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(${TGT} ${LIBRARIES} )

    install(TARGETS ${TGT} DESTINATION bin)
endforeach()




