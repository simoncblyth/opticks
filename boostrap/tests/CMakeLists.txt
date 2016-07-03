cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BoostRapTest)
project(${name})

#file(GLOB TEST_CC_SRCS "B*Test.cc")

set(TEST_SOURCES

   BFooTest.cc
   BDirTest.cc
   BTimeTest.cc
   BMapTest.cc
   BListTest.cc
   BRegexTest.cc
   BStrTest.cc
   BTimerTest.cc
   BBitTest.cc
   BoostLogTest.cc 
   BLogTest.cc
   BFileTest.cc
   BBufSpecTest.cc
)



foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} BoostRap pthread)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


