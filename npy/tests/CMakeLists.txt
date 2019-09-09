cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name NPYTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


set(_TESTS_FAILING_TO_INVESTIGATE_WHEN_IN_CONTEXT

_TorchStepNPYTest.cc

)


set(_TEST_SOURCES

FabStepTest.cc
NPriTest.cc

NXformTest.cc 

EnvTest.cc
NPYBaseTest.cc
NSliceTest.cc
NPYSpecTest.cc
NPYSpecListTest.cc
NPYListTest.cc
NPYMetaTest.cc
NGeoTestConfigTest.cc

GLMPrintTest.cc
GLMFormatTest.cc

ViewNPYTest.cc


NStateTest.cc
ReportTest.cc
IndexTest.cc
CountsTest.cc



TypesTest.cc
readFlagsTest.cc

NSensorListTest.cc
NGunConfigTest.cc

NQuadTest.cc
NBoundingBoxTest.cc
NPartTest.cc
NSpectralTest.cc

NTorusTest.cc
NHyperboloidTest.cc
NCubicTest.cc
NSphereTest.cc
NZSphereTest.cc
NBoxTest.cc
NBox2Test.cc
NNode2Test.cc

NSlabTest.cc
NPlaneTest.cc
NCylinderTest.cc
NDiscTest.cc
NConeTest.cc
NConvexPolyhedronTest.cc
NConvexPolyhedronSegmentTest.cc

NNodeTest.cc
NNodeDumpTest.cc
NNodeDumpMinimalTest.cc
NNodeTreeTest.cc
NNodePointsTest.cc

NTesselateTest.cc


NEmitPhotonsNPYTest.cc


_BoundariesNPYTest.cc
_RecordsNPYTest.cc
NTrianglesNPYTest.cc
PhotonsNPYTest.cc

NCIETest.cc
NPYTest.cc
NPYLoadTest.cc
NPY2Test.cc
NPY3Test.cc
NPYminmaxTest.cc
NPYreshapeTest.cc

NLoadTest.cc


DummyPhotonsNPYTest.cc

NMarchingCubesNPYTest.cc
NCSGTest.cc
NCSG2Test.cc
NCSGLoadTest.cc
NCSGSaveTest.cc
NCSGScanTest.cc
NCSGListTest.cc
NEmitConfigTest.cc
NCSGIntersectTest.cc
NCSGRoundTripTest.cc
NScanTest.cc

NCSGPolygonizeTest.cc 

EllipseTest.cc
NSolidTest.cc
NSolid2Test.cc
NTreeJUNOTest.cc


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
NSnapConfigTest.cc
NRngDiffuseTest.cc


NTreeBuilderTest.cc
NTreePositiveTest.cc
NTreeBalanceTest.cc
NTreeAnalyseTest.cc
NGPUTest.cc

NNodeCollectorTest.cc
NGridTest.cc



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
       NSceneConfigTest.cc
       #NYMathTest.cc
       #NYShapeTest.cc
       NSDFTest.cc
       NSceneLoadTest.cc
       NSceneMeshTest.cc
       NYJSONTest.cc
       NJSTest.cc
       NMetaTest.cc



)
endif()



set( _TEST_SOURCES_SKIPPED

       NSceneTest.cc


)




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


# override for faster cycle
#set(TEST_SOURCES
#)




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

    #add_dependencies(check ${TGT})
    target_link_libraries(${TGT} NPY) 
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()




# promote some python scripts to the standard binary location
set(PY_TEST_SOURCES
)

install(FILES ${PY_TEST_SOURCES} DESTINATION lib 
        PERMISSIONS 
            WORLD_READ GROUP_READ OWNER_READ 
            WORLD_WRITE GROUP_WRITE OWNER_WRITE 
            WORLD_EXECUTE GROUP_EXECUTE OWNER_EXECUTE 
)




