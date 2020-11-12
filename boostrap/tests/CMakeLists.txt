cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name BoostRapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(TEST_SOURCES

   BFooTest.cc
   BDirTest.cc

   BMapTest.cc
   MapTest.cc

   BListTest.cc
   BParametersTest.cc
   BTxtTest.cc
   BPropNamesTest.cc

   BHexTest.cc
   BRegexTest.cc
   BStrTest.cc
   BLogTest.cc
   BPLOGTest.cc
   BFileTest.cc
   BPathTest.cc
   BBufSpecTest.cc
   BEnvTest.cc
   BOpticksTest.cc
   BOpticksResourceTest.cc
   BResourceTest.cc
   BOpticksKeyTest.cc
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

   BConverterTest.cc 
   BRAP_OPTICKS_LOG_Test.cc


   BTimeKeeperTest.cc
   BTimeStampTest.cc
   BTimesTest.cc
   BTimesTableTest.cc


)



if(BoostAsio_FOUND)  
   list(APPEND TEST_SOURCES 
        BListenUDPTest.cc     
   )
endif()



#[=[
BTimeTest.cc requires boost components : thread timer, linking gives 

    CMakeFiles/BTimeTest.dir/BTimeTest.cc.o: In function `boost::this_thread::sleep(boost::posix_time::ptime const&)':
    /usr/include/boost/thread/pthread/thread_data.hpp:247: undefined reference to `boost::this_thread::hiden::sleep_until(timespec const&)'

#]=]


set(_TEST_SOURCES
   BTimeTest.cc
   boost_random_Test.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

    target_link_libraries(${TGT} BoostRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

