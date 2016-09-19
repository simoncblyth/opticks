cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name SysRapTest)
project(${name})

set(TEST_SOURCES

   SArgsTest.cc
   SEnvTest.cc
   SSysTest.cc
   SProcTest.cc
   SDigestTest.cc
   OpticksCMakeConfigTest.cc
   PLogTest.cc 
   PLOG_Test.cc 
   SYSRAP_LOG_Test.cc 
   SYSRAP_LOG_FileAppenderTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} SysRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


