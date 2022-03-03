cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name G4OKTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


set(TESTS
    G4OKTest.cc
    G4OpticksHitTest.cc

)

foreach(SRC ${TESTS})

    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} G4OK)
    install(TARGETS ${TGT} DESTINATION lib)

    add_test(${name}.${TGT} ${TGT})

endforeach()


find_package(PMTSim CONFIG)

set(PMTSIM_TEST_SOURCES
   G4OKPMTSimTest.cc 
)


foreach(TEST_CC_SRC ${PMTSIM_TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    set(testname ${name}.${TGT})
    add_test(${testname} ${TGT})

    target_link_libraries(${TGT} PUBLIC G4OK )

    if(PMTSim_FOUND)
       target_link_libraries( ${TGT} PUBLIC Opticks::PMTSim )
       target_compile_definitions( ${TGT} PUBLIC WITH_PMTSIM )
    endif()

    install(TARGETS ${TGT} DESTINATION lib)
endforeach()







