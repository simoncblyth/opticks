cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPYTest)
project(${name})



#file(GLOB TEST_SOURCES "*Test.cc")

set(TEST_OK

LogTest.cc
EnvTest.cc
NPYBaseTest.cc
NSliceTest.cc
NPYTest.cc
NPYSpecTest.cc

GLMPrintTest.cc
GLMFormatTest.cc

ViewNPYTest.cc
MultiViewNPYTest.cc
NStateTest.cc
ParametersTest.cc
ReportTest.cc
LookupTest.cc
MapTest.cc
IndexTest.cc
NPropNamesTest.cc
CountsTest.cc

TimerTest.cc
TimesTableTest.cc
TimesTest.cc
NTxtTest.cc

TypesTest.cc
readFlagsTest.cc

NSensorListTest.cc
NGunConfigTest.cc
)


set(TEST_SOURCES

NQuadTest.cc
NBoundingBoxTest.cc
NPartTest.cc
NSpectralTest.cc
NSphereTest.cc
NTesselateTest.cc

)


set(FUTURE_TEST_SOURCES

_BoundariesNPYTest.cc
_RecordsNPYTest.cc
_TorchStepNPYTest.cc

G4StepNPYTest.cc
HitsNPYTest.cc
NTrianglesNPYTest.cc
PhotonsNPYTest.cc
SequenceNPYTest.cc

)


foreach(TEST_SOURCE ${TEST_SOURCES})

    get_filename_component(TGT ${TEST_SOURCE} NAME_WE)
    add_executable(${TGT} ${TEST_SOURCE})


    set(testname ${name}.${TGT})  

    # see cmakex-
    add_test(${testname} ${TGT})

    if(WIN32)
       # https://cmake.org/pipermail/cmake/2010-December/041176.html
       set(PATH_STRING   "PATH=${CMAKE_INSTALL_PREFIX}/lib;$ENV{PATH}")
       STRING(REPLACE "\\;" ";" PATH_STRING "${PATH_STRING}")
       STRING(REPLACE ";" "\\;" PATH_STRING "${PATH_STRING}")
       set_property(TEST ${testname} PROPERTY ENVIRONMENT ${PATH_STRING})
    endif(WIN32)

    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} 
               ${LIBRARIES} 
               NPY
    )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

