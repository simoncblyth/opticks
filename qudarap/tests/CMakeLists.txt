cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name QUDARapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   


set(TEST_SOURCES

   QTexRotateTest.cc
   QRngTest.cc
   QScintTest.cc
   QCerenkovIntegralTest.cc
   QCerenkovTest.cc
   QPolyTest.cc
   QSimTest.cc
   QBndTest.cc 
   QPrdTest.cc 
   QOpticalTest.cc 
   QPropTest.cc 
   QSeedTest.cc 
   QEventTest.cc 
   QSimWithEventTest.cc 
   QUTest.cc
   QTexMakerTest.cc
   QTexLookupTest.cc
   QBufTest.cc

)

set(OTHER_TEST_SOURCES
   QCKTest.cc
)


foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    add_test(${name}.${TGT} ${TGT})
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


