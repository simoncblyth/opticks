cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name BoostRapTest)
project(${name})

#file(GLOB TEST_CC_SRCS "B*Test.cc")

set(TEST_SOURCES_OK

   BFooTest.cc
   BDirTest.cc
   BTimeTest.cc
   BMapTest.cc
   BListTest.cc
   BSysTest.cc
   BRegexTest.cc
   BDigestTest.cc
   BStrTest.cc
   BTimerTest.cc
   BBitTest.cc
   BoostLogTest.cc 
   PLogTest.cc 
   BLogTest.cc
   BFileTest.cc
)


set(TEST_SOURCES

   BBufSpecTest.cc

)


set(FAIL_TO_LINK_ON_WINDOWS
   BJsonTest.cc
)



foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} BoostRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


