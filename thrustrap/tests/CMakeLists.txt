cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name ThrustRapTest)
project(${name})

file(GLOB TEST_CC_SRCS "*Test.cc")

foreach(TEST_CC_SRC ${TEST_CC_SRCS})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)

    add_executable(${TGT} ${TEST_CC_SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})


    target_link_libraries(${TGT} 
            ${LIBRARIES} 
            ThrustRap
    )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



file(GLOB TEST_CU_SRCS "*Test.cu")


set(XOPT --disable-warnings)
foreach(TEST_CU_SRC ${TEST_CU_SRCS})
    get_filename_component(TGT ${TEST_CU_SRC} NAME_WE)
    cuda_add_executable(${TGT} ${TEST_CU_SRC} OPTIONS ${XOPT})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} 
            ${LIBRARIES} 
            ThrustRap
    )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



