2017-10-23-several-tests-take-longer-than-1sec
==================================================


Top 10 offenders
-------------------

::

    simon:opticks blyth$ opticks-ts
    255/255 Test #255: okg4Test.OKG4Test ............................   Passed   19.68 sec
    246/255 Test #246: cfg4Test.CG4Test .............................   Passed   11.59 sec
    250/255 Test #250: cfg4Test.CCollectorTest ......................   Passed    9.47 sec
    235/255 Test #235: OKOPTest.OpTest ..............................   Passed    7.55 sec
    237/255 Test #237: OKTest.OKTest ................................   Passed    7.55 sec
    127/255 Test #127: NPYTest.NSceneMeshTest .......................   Passed    6.47 sec
    240/255 Test #240: OKTest.VizTest ...............................   Passed    6.43 sec
    228/255 Test #228: OptiXRapTest.ORayleighTest ...................   Passed    6.31 sec
    232/255 Test #232: OKOPTest.OpSeederTest ........................   Passed    6.18 sec
    227/255 Test #227: OptiXRapTest.OInterpolationTest ..............   Passed    4.83 sec


opticks-ts
---------------

::

    simon:opticks blyth$ opticks-t
    Mon Oct 23 16:13:35 CST 2017
    Test project /usr/local/opticks/build
            Start   1: SysRapTest.SArTest
      1/255 Test   #1: SysRapTest.SArTest ...........................   Passed    0.00 sec
    ...
    252/255 Test #252: cfg4Test.OpRayleighTest ......................   Passed    1.29 sec
            Start 253: cfg4Test.CGROUPVELTest
    253/255 Test #253: cfg4Test.CGROUPVELTest .......................   Passed    0.47 sec
            Start 254: cfg4Test.CMakerTest
    254/255 Test #254: cfg4Test.CMakerTest ..........................   Passed    0.02 sec
            Start 255: okg4Test.OKG4Test
    255/255 Test #255: okg4Test.OKG4Test ............................   Passed   19.68 sec

    100% tests passed, 0 tests failed out of 255

    Total Test time (real) = 133.23 sec
    Mon Oct 23 16:15:49 CST 2017
    opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/ctest.log
    simon:opticks blyth$ 

    simon:opticks blyth$ opticks-
    simon:opticks blyth$ opticks-ts
      5/255 Test   #5: SysRapTest.SSysTest ..........................   Passed    1.05 sec
    103/255 Test #103: NPYTest.NScanTest ............................   Passed    3.09 sec
    121/255 Test #121: NPYTest.NDualContouringSampleTest ............   Passed    1.27 sec
    127/255 Test #127: NPYTest.NSceneMeshTest .......................   Passed    6.47 sec
    180/255 Test #180: GGeoTest.GMakerTest ..........................   Passed    3.65 sec
    192/255 Test #192: GGeoTest.GSceneTest ..........................   Passed    1.19 sec
    194/255 Test #194: AssimpRapTest.AssimpRapTest ..................   Passed    3.02 sec
    196/255 Test #196: AssimpRapTest.AssimpGGeoTest .................   Passed    1.83 sec
    199/255 Test #199: OpticksGeometryTest.OpenMeshRapTest ..........   Passed    1.85 sec
    226/255 Test #226: OptiXRapTest.OEventTest ......................   Passed    1.11 sec
    227/255 Test #227: OptiXRapTest.OInterpolationTest ..............   Passed    4.83 sec
    228/255 Test #228: OptiXRapTest.ORayleighTest ...................   Passed    6.31 sec
    231/255 Test #231: OKOPTest.OpIndexerTest .......................   Passed    1.04 sec
    232/255 Test #232: OKOPTest.OpSeederTest ........................   Passed    6.18 sec
    235/255 Test #235: OKOPTest.OpTest ..............................   Passed    7.55 sec
    237/255 Test #237: OKTest.OKTest ................................   Passed    7.55 sec
    238/255 Test #238: OKTest.OTracerTest ...........................   Passed    1.34 sec
    240/255 Test #240: OKTest.VizTest ...............................   Passed    6.43 sec
    246/255 Test #246: cfg4Test.CG4Test .............................   Passed   11.59 sec
    250/255 Test #250: cfg4Test.CCollectorTest ......................   Passed    9.47 sec
    251/255 Test #251: cfg4Test.CInterpolationTest ..................   Passed    1.39 sec
    252/255 Test #252: cfg4Test.OpRayleighTest ......................   Passed    1.29 sec
    255/255 Test #255: okg4Test.OKG4Test ............................   Passed   19.68 sec
    Total Test time (real) = 133.23 sec
    simon:opticks blyth$ 










