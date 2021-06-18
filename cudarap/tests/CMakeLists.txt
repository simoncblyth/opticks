cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CUDARapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   


set(TEST_SOURCES
   LaunchSequenceTest.cc
   cuRANDWrapperTest.cc
   curand_aligned_host.cc
   CUDARapVersionTest.cc
   CDeviceTest.cc
   CRngTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})
    target_link_libraries(${TGT} CUDARap )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


set(TEST_SOURCES_CU
   cudaGetDevicePropertiesTest.cu
   cfloat4x4Test.cu
)

foreach(CU ${TEST_SOURCES_CU})
    get_filename_component(TGT ${CU} NAME_WE)
    cuda_add_executable(${TGT} ${CU} OPTIONS)
    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

    target_link_libraries(${TGT} CUDARap )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


