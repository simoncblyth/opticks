cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name NPYTest)
project(${name})


set(_TESTS_FAILING_TO_INVESTIGATE_WHEN_IN_CONTEXT

_TorchStepNPYTest.cc
FabStepTest.cc

)


set(_TEST_SOURCES

EnvTest.cc
NPYBaseTest.cc
NSliceTest.cc
NPYSpecTest.cc

GLMPrintTest.cc
GLMFormatTest.cc

ViewNPYTest.cc


NStateTest.cc
NParametersTest.cc
ReportTest.cc
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

NQuadTest.cc
NBoundingBoxTest.cc
NPartTest.cc
NSpectralTest.cc

NSphereTest.cc
NZSphereTest.cc
NBoxTest.cc
NSlabTest.cc
NPlaneTest.cc
NCylinderTest.cc
NConeTest.cc
NConvexPolyhedronTest.cc

NNodeTest.cc

NTesselateTest.cc



_BoundariesNPYTest.cc
_RecordsNPYTest.cc
NTrianglesNPYTest.cc
PhotonsNPYTest.cc

NCIETest.cc
NPYTest.cc
NPY2Test.cc
NLoadTest.cc


DummyPhotonsNPYTest.cc

NMarchingCubesNPYTest.cc
NCSGTest.cc
NCSGLoadTest.cc
NCSGScanTest.cc

NCSGPolygonizeTest.cc 




NOctNodeTest.cc

NFieldCacheTest.cc
NGeneratorTest.cc
mortonlibTest.cc
NGrid3Test.cc
NField3Test.cc
NTreeToolsTest.cc

NGLMExtTest.cc


NLoadCheck.cc
NGLMTest.cc
NCSGDeserializeTest.cc
MultiViewNPYTest.cc
HitsNPYTest.cc
G4StepNPYTest.cc


NTrisTest.cc




)

set(TEST_SOURCES

NPYConfigTest.cc
NBBoxTest.cc

NOpenMeshTest.cc
NOpenMeshCfgTest.cc
NOpenMeshFindTest.cc
NOpenMeshCombineTest.cc
NPolygonizerTest.cc
NCSGBSPTest.cc
NuvTest.cc

)

list(APPEND TEST_SOURCES ${_TEST_SOURCES})





if(NonExisting_FOUND)
   list(APPEND TEST_SOURCES  NNonExistingTest.cc)
endif()

if(ImplicitMesher_FOUND)
   list(APPEND TEST_SOURCES  NImplicitMesherTest.cc)
endif()

if(DualContouringSample_FOUND)
   list(APPEND TEST_SOURCES  
       NDualContouringSampleTest.cc
       NFieldGrid3Test.cc
)
endif()


if(YoctoGL_FOUND)
   list(APPEND TEST_SOURCES  
       NSceneTest.cc
       NYMathTest.cc
       NYShapeTest.cc
)
endif()






set(_TEST_SOURCES_WITH_LINK_ERRORS_FROM_FWD_NPY_DECL_IN_NGLMEXT_HPP

NLoadCheck.cc
NGLMTest.cc
NCSGDeserializeTest.cc
NBBoxTest.cc
MultiViewNPYTest.cc
HitsNPYTest.cc
G4StepNPYTest.cc

)


set(_TEST_SOURCES




SequenceNPYTest.cc
NConstructorTest.cc

)




foreach(TEST_SOURCE ${TEST_SOURCES})

    get_filename_component(TGT ${TEST_SOURCE} NAME_WE)
    add_executable(${TGT} ${TEST_SOURCE})

    set(testname ${name}.${TGT})  

    # see cmakex-
    add_test(${testname} ${TGT})

    if(WIN32)
       # MAYBE AVOID THIS MESS JUST BY SETTING PATH FOR VS PROCESS ? 
       # https://cmake.org/pipermail/cmake/2010-December/041176.html
       set(PATH_STRING   "PATH=${CMAKE_INSTALL_PREFIX}/lib;$ENV{PATH}")
       STRING(REPLACE "\\;" ";" PATH_STRING "${PATH_STRING}")
       STRING(REPLACE ";" "\\;" PATH_STRING "${PATH_STRING}")
       set_property(TEST ${testname} PROPERTY ENVIRONMENT ${PATH_STRING})
    endif(WIN32)

    add_dependencies(check ${TGT})
    target_link_libraries(${TGT} ${LIBRARIES} NPY) 
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

