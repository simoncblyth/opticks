cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name QUDARapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(INSTALLED_SCRIPTS
   QTestRunner.sh 
)
install(PROGRAMS ${INSTALLED_SCRIPTS} DESTINATION bin)




set(TEST_SOURCES

   QTexRotateTest.cc

   QRngTest.cc
   QScintTest.cc
   QCerenkovIntegralTest.cc
   QPolyTest.cc
   QSimTest.cc
   QSimDescTest.cc
   QOpticalTest.cc 
   QPropTest.cc 
   QEventTest.cc 
   QEvent_Lifecycle_Test.cc
   QSim_Lifecycle_Test.cc
   QSimWithEventTest.cc 
   QSimCreateTest.cc 
   QUTest.cc
   QTexMakerTest.cc
   QTexLookupTest.cc
   QBufTest.cc
   QMultiFilmTest.cc


   QCurandStateMonolithicTest.cc
   QCurandStateTest.cc

   qrng_test.cc
)



## THESE NEED REJIG : THEY FAIL FOR LACK OF DEPENDENCIES 
set(KNOWN_FAIL_TEST_SOURCES
   QCerenkovTest.cc
   QBndTest.cc 
)

set(OTHER_TEST_SOURCES
   QCKTest.cc
)


find_program(BASH_EXECUTABLE NAMES bash REQUIRED)
message(STATUS "BASH_EXECUTABLE : ${BASH_EXECUTABLE}")

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} QUDARap )
    install(TARGETS ${TGT} DESTINATION lib)

    #add_test(${name}.${TGT} ${TGT})
    add_test(
       NAME ${name}.${TGT} 
       COMMAND bash QTestRunner.sh ${TGT}
       #COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/QTestRunner.sh ${TGT}
    )

endforeach()


#[=[
#]=]
foreach(SRC ${KNOWN_FAIL_TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} QUDARap )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()




set(TEST_SOURCES_CU
)

foreach(CU ${TEST_SOURCES_CU})
    get_filename_component(TGT ${CU} NAME_WE)
    cuda_add_executable(${TGT} ${CU} OPTIONS)
    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} QUDARap )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



## HMM: switch to Custom4 (not getting that indirectly via PMTSim) 
find_package(PMTSim_standalone CONFIG)

if(PMTSim_standalone_FOUND)

    set(PMTSIM_TEST_SOURCES
        QPMTTest.cc 
    )

    foreach(SRC ${PMTSIM_TEST_SOURCES})
        get_filename_component(TGT ${SRC} NAME_WE)
        add_executable(${TGT} ${SRC})

        set(testname ${name}.${TGT})
        add_test(${testname} ${TGT})

        target_link_libraries(${TGT} QUDARap )
        target_link_libraries( ${TGT} Opticks::PMTSim_standalone )
        target_compile_definitions( ${TGT} PRIVATE WITH_PMTSIM )

        install(TARGETS ${TGT} DESTINATION lib)
    endforeach()

else()
    message(STATUS "${name} PMTSim_standalone_FOUND:${PMTSim_standalone_FOUND}  " )

endif()




