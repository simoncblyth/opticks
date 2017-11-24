cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BoostRapTest)
project(${name})

#file(GLOB TEST_CC_SRCS "B*Test.cc")

set(TEST_SOURCES

   BFooTest.cc
   BDirTest.cc
   BMapTest.cc
   BHexTest.cc
   MapTest.cc
   BListTest.cc
   BRegexTest.cc
   BStrTest.cc
   BTimerTest.cc
   BBitTest.cc
   BLogTest.cc
   BFileTest.cc
   BBufSpecTest.cc
   BEnvTest.cc
   BOpticksResourceTest.cc
   BDynamicDefineTest.cc
   BOpticksEventTest.cc
   LexicalCastTest.cc
   BBisectTest.cc
   BConfigTest.cc
   BBndTest.cc
   BConfigConstituentTest.cc
   BRngTest.cc

   boost_random_demo_Test.cc
   boost_random_minimal_Test.cc

)


# BTimeTest.cc requires boost components : thread timer 
set(TEST_SOURCES_NOT

   BTimeTest.cc
   boost_random_Test.cc

)


foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} BoostRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


