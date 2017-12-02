cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name CUDARapTest)
project(${name})

#message("${name}.INCLUDE_DIRS : ${INCLUDE_DIRS}")
message("${name}.LIBRARIES    : ${LIBRARIES}")

set(TEST_SOURCES
   LaunchSequenceTest.cc
   cuRANDWrapperTest.cc
   curand_aligned_host.cc
   
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})
    target_link_libraries(${TGT} ${LIBRARIES}  )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


#file(GLOB TEST_CU_SRCS "*Test.cu")

set(TEST_SOURCES_CU
   cudaGetDevicePropertiesTest.cu
)

foreach(CU ${TEST_SOURCES_CU})
    get_filename_component(TGT ${CU} NAME_WE)
    cuda_add_executable(${TGT} ${CU} OPTIONS)
    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


