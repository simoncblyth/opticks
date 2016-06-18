cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPYTest)
project(${name})



#file(GLOB TEST_SOURCES "*Test.cc")

set(TEST_SOURCES

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
)


set(FUTURE_TEST_SOURCES

_BoundariesNPYTest.cc
_RecordsNPYTest.cc
_TorchStepNPYTest.cc
CountsTest.cc
G4StepNPYTest.cc
HitsNPYTest.cc
IndexTest.cc
MapTest.cc
NBoundingBoxTest.cc
NPartTest.cc
NPropNamesTest.cc
NQuadTest.cc
NSensorListTest.cc
NSpectralTest.cc
NSphereTest.cc
NTesselateTest.cc
NTrianglesNPYTest.cc
NTxtTest.cc
PhotonsNPYTest.cc
readFlagsTest.cc
ReportTest.cc
SequenceNPYTest.cc
TimerTest.cc
TimesTableTest.cc
TimesTest.cc
TypesTest.cc

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

