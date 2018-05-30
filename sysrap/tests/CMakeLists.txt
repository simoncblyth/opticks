cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name SysRapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(TEST_SOURCES

   SOKConfTest.cc
   SArTest.cc
   SArgsTest.cc
   STimesTest.cc
   SEnvTest.cc
   SSysTest.cc
   STrancheTest.cc
   SVecTest.cc
   SMapTest.cc
   SSeqTest.cc
   SProcTest.cc
   SBase36Test.cc
   SSortKVTest.cc
   SPairVecTest.cc
   SDigestTest.cc
   PLogTest.cc 
   PLOG_Test.cc 
   SYSRAP_LOG_Test.cc 
   SYSRAP_LOG_FileAppenderTest.cc

   SYSRAP_OPTICKS_LOG_Test.cc

   OpticksCSGTest.cc
   hash_define_without_value.cc
   SDirectTest.cc 
   S_freopen_redirect_test.cc

)

set(_TEST_SOURCES
   UnwindTest.cc
   OpticksCMakeConfigTest.cc
)


set(OBJC_TEST_SOURCES
)

set(_OBJC_TEST_SOURCES

   NSLogTest_m.m
   NSLogTest_mm.mm
)




foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

    target_link_libraries(${TGT} SysRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


#foreach(SRC ${OBJC_TEST_SOURCES})
#    get_filename_component(TGT ${SRC} NAME_WE)
#    add_executable(${TGT} ${SRC})
#
#    add_test(${name}.${TGT} ${TGT})
#    #add_dependencies(check ${TGT})
#
#    target_link_libraries(${TGT} ${LIBRARIES} "-framework Foundation")
#    install(TARGETS ${TGT} DESTINATION lib)
#endforeach()



