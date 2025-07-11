cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(name SysRapTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(INSTALLED_SCRIPTS
   STestRunner.sh
   sseq_index_test.sh
   sseq_index_test.py

   SGLFW_SOPTIX_Scene_test.sh
   ssst.sh
)

install(PROGRAMS ${INSTALLED_SCRIPTS}   DESTINATION bin)

set(TEST_SOURCES

   PythonImportTest.cc
   SOKConfTest.cc
   SArTest.cc
   SArrTest.cc
   SArgsTest.cc
   STimesTest.cc
   SEnvTest.cc
   SSysTest.cc
   SSys2Test.cc
   SSys3Test.cc
   SStrTest.cc
   SPathTest.cc
   STrancheTest.cc
   SVecTest.cc
   SNameVecTest.cc
   SMapTest.cc
   SCountTest.cc
   SSeqTest.cc
   SBase36Test.cc
   SSortKVTest.cc
   SPairVecTest.cc
   SDigestTest.cc
   SDigestNPTest.cc

   SCFTest.cc


   PLogTest.cc
   SLOG_Test.cc
   SLOGTest.cc
   SYSRAP_LOG_Test.cc
   SYSRAP_LOG_FileAppenderTest.cc

   SYSRAP_OPTICKS_LOG_Test.cc
   SYSRAP_OPTICKS_LOG_NULL_Test.cc
   SOPTICKS_LOG_Test.cc

   OPTICKS_LOG_Test.cc
   sLOG_MACRO_Test.cc
   SLOG_exename_Test.cc
   SLOG_Banner_Test.cc

   reallocTest.cc
   OpticksCSGTest.cc
   hash_define_without_value.cc
   SDirectTest.cc
   S_freopen_redirect_test.cc
   S_get_option_Test.cc

   SIdTest.cc

   ArrayTest.cc
   SBacktraceTest.cc
   SStackFrameTest.cc
   SGDMLTest.cc

   SSetTest.cc
   STimeTest.cc
   SASCIITest.cc
   SAbbrevTest.cc
   SPPMTest.cc
   SColorTest.cc
   SPackTest.cc
   SBitTest.cc
   SBitSetTest.cc
   SEnabledTest.cc
   SBitFromStringTest.cc
   SRandTest.cc
   SOpticksTest.cc
   SOpticksKeyTest.cc


   CheckGeoTest.cc

   SGLMTest.cc

   SConstantTest.cc
   SLabelCacheTest.cc

   CastTest.cc

   SOpticksVersionNumberTest.cc
   SRngTest.cc
   SDirTest.cc
   SDiceTest.cc

   stranTest.cc
   stranRotateTest.cc

   SCenterExtentGenstepTest.cc
   SFrameGenstep_MakeCenterExtentGensteps_Test.cc

   SEventTest.cc
   SThetaCutTest.cc
   SPhiCutTest.cc
   scanvasTest.cc

   OpticksPhotonTest.cc
   SUTest.cc
   SEventConfigTest.cc
   SFrameConfigTest.cc
   SGeoConfigTest.cc



   SEvtTest.cc
   SEvtLoadTest.cc
   SEvt__UU_BURN_Test.cc

   SEvt_Lifecycle_Test.cc
   SEvt__HasInputPhoton_Test.cc
   SEvt_AddEnvMeta_Test.cc

   SNameTest.cc

   SProfTest.cc

   SEvt_test.cc
   sseq_index_test.cc

   ssys_test.cc
   srng_test.cc
   sbuild_test.cc

   SCurandStateMonolithicTest.cc

)


set( OPTIX_TEST_SOURCES

   SGLFW_SOPTIX_Scene_test.cc

)

set(CUDART_TEST_SOURCES

   scontext_test.cc
)


set(RETIRED_TEST_SOURCES
   SProcTest.cc
   SRngSpecTest.cc
)



set(REPORT_SOURCES
   sreport.cc
)




set(BASH_RUN_TEST_SOURCES
    SOpticksResourceTest.cc
    SEnvTest_PASS.cc
    SSimTest.cc
    SBndTest.cc

    SSceneLoadTest.cc
)

if(NLJSON_FOUND)
   list(APPEND TEST_SOURCES SMetaTest.cc)
endif()


if(WITH_SIMG)
   list(APPEND TEST_SOURCES SIMGTest.cc)
endif()

if(WITH_STTF)
   list(APPEND TEST_SOURCES STTFTest.cc)
endif()




set(_TEST_SOURCES
   UnwindTest.cc
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
    target_link_libraries(${TGT} SysRap)

    target_include_directories( ${TGT} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../inc>
    )

    install(TARGETS ${TGT} DESTINATION lib)
    add_test(${name}.${TGT} ${TGT})
endforeach()

foreach(SRC ${CUDART_TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} SysRap CUDA::cudart)


    target_include_directories( ${TGT} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../inc>
    )

    install(TARGETS ${TGT} DESTINATION lib)
    add_test(${name}.${TGT} ${TGT})
endforeach()



find_package(OpticksGLEW REQUIRED MODULE)
find_package(OpticksGLFW REQUIRED MODULE)


foreach(SRC ${OPTIX_TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT}
        SysRap
        CUDA::cudart
        Opticks::OpticksGLEW
        Opticks::OpticksGLFW
    )

    target_include_directories( ${TGT} PRIVATE
        ${OpticksOptiX_INCLUDE}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../inc>
    )

    install(TARGETS ${TGT} DESTINATION lib)
    #add_test(${name}.${TGT} ${TGT})
    add_test(
       NAME ${name}.${TGT}
       COMMAND bash STestRunner.sh ${TGT}
    )
endforeach()




foreach(SRC ${REPORT_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} SysRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()









set(TEST_COLORS
    red
    green
    blue
)

foreach(COLOR ${TEST_COLORS})
    set(TGT SEnvTest)
    set(TEST ${name}.${TGT}.${COLOR})
    add_test(${TEST} ${TGT} SEnvTest_C --info )  # arguments for the executable, the envvar prefix and debug level
    set_tests_properties(${TEST} PROPERTIES ENVIRONMENT "SEnvTest_COLOR=${COLOR}")
endforeach()

#[=[
Debug these "fabricated" tests with::

   ctest -R SysRapTest.SEnvTest. -V

#]=]


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



find_program(BASH_EXECUTABLE NAMES bash REQUIRED)
message(STATUS "BASH_EXECUTABLE : ${BASH_EXECUTABLE}")

foreach(SRC ${BASH_RUN_TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} SysRap)
    install(TARGETS ${TGT} DESTINATION lib)
    add_test(
       NAME ${name}.${TGT}
       COMMAND bash STestRunner.sh ${TGT}
       #COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/STestRunner.sh ${TGT}
    )
    #[=[
    Try to be more portable by getting bash and Runner off the PATH
    #]=]
endforeach()

