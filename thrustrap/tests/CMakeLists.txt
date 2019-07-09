cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name ThrustRapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(TEST_CC_SRCS
    TCURANDTest.cc
)

foreach(TEST_CC_SRC ${TEST_CC_SRCS})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)

    add_executable(${TGT} ${TEST_CC_SRC})

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ThrustRap) 
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


#file(GLOB TEST_CU_SRCS "*Test.cu")
set(TEST_CU_SRCS

CBufSpecTest.cu
TBufTest.cu
TRngBufTest.cu
expandTest.cu
iexpandTest.cu
issue628Test.cu
printfTest.cu
repeated_rangeTest.cu
strided_rangeTest.cu
strided_repeated_rangeTest.cu

float2intTest.cu 
thrust_curand_estimate_pi.cu
thrust_curand_printf.cu
thrust_curand_printf_redirect.cu
thrust_curand_printf_redirect2.cu
TBuf4x4Test.cu

)

set(_TEST_CU_SRCS
)


foreach(TEST_CU_SRC ${TEST_CU_SRCS})
    get_filename_component(TGT ${TEST_CU_SRC} NAME_WE)
    cuda_add_executable(${TGT} ${TEST_CU_SRC} OPTIONS )

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})
    target_link_libraries(${TGT} ThrustRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



