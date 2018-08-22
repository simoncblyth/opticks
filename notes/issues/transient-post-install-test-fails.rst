transient-post-install-test-fails
===================================

Many test fails from lack of geocache on first test::

    FAILS:
      74 /111 Test #74 : NPYTest.NPYreshapeTest                        ***Exception: SegFault         0.01   
      9  /49  Test #9  : GGeoTest.GMaterialLibTest                     ***Exception: Child aborted    0.02   
      12 /49  Test #12 : GGeoTest.GScintillatorLibTest                 ***Exception: Child aborted    0.01   
      15 /49  Test #15 : GGeoTest.GBndLibTest                          ***Exception: Child aborted    0.02   
      16 /49  Test #16 : GGeoTest.GBndLibInitTest                      ***Exception: Child aborted    0.02   
      27 /49  Test #27 : GGeoTest.GPartsTest                           ***Exception: Child aborted    0.02   
      29 /49  Test #29 : GGeoTest.GPmtTest                             ***Exception: Child aborted    0.02   
      30 /49  Test #30 : GGeoTest.BoundariesNPYTest                    ***Exception: Child aborted    0.02   
      31 /49  Test #31 : GGeoTest.GAttrSeqTest                         ***Exception: Child aborted    0.02   
      35 /49  Test #35 : GGeoTest.GGeoLibTest                          ***Exception: Child aborted    0.01   
      36 /49  Test #36 : GGeoTest.GGeoTest                             ***Exception: Child aborted    0.02   
      37 /49  Test #37 : GGeoTest.GMakerTest                           ***Exception: Child aborted    0.02   
      44 /49  Test #44 : GGeoTest.GSurfaceLibTest                      ***Exception: Child aborted    0.01   
      46 /49  Test #46 : GGeoTest.NLookupTest                          ***Exception: Child aborted    0.01   
      47 /49  Test #47 : GGeoTest.RecordsNPYTest                       ***Exception: Child aborted    0.02   
      48 /49  Test #48 : GGeoTest.GSceneTest                           ***Exception: Child aborted    0.02   
      49 /49  Test #49 : GGeoTest.GMeshLibTest                         ***Exception: Child aborted    0.02   
      3  /3   Test #3  : OpticksGeoTest.OpenMeshRapTest                ***Exception: SegFault         0.95   
      16 /18  Test #16 : OptiXRapTest.ORayleighTest                    ***Exception: Child aborted    1.61   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         ***Exception: Child aborted    1.45   
      2  /5   Test #2  : OKTest.OKTest                                 ***Exception: Child aborted    1.92   
      1  /22  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: Child aborted    0.29   
      2  /22  Test #2  : CFG4Test.CMaterialTest                        ***Exception: Child aborted    0.28   
      3  /22  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    0.75   
      4  /22  Test #4  : CFG4Test.CGDMLDetectorTest                    ***Exception: Child aborted    0.68   
      5  /22  Test #5  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.67   
      6  /22  Test #6  : CFG4Test.CG4Test                              ***Exception: Child aborted    0.71   
      17 /22  Test #17 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    0.69   
      19 /22  Test #19 : CFG4Test.CGROUPVELTest                        ***Exception: Child aborted    0.27   
      22 /22  Test #22 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.69   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    0.72   
    epsilon:opticks-test-copy blyth$ 


Axel got::

    totals  42  / 359 


    FAILS:
      17 /29  Test #17 : BoostRapTest.BOpticksKeyTest                  ***Exception: SegFault         0.21   
      9  /23  Test #9  : OpticksCoreTest.OpticksCfgTest                ***Exception: SegFault         0.15   
      10 /23  Test #10 : OpticksCoreTest.OpticksTest                   ***Exception: SegFault         0.20   
      4  /50  Test #4  : GGeoTest.GBufferTest                          ***Exception: SegFault         0.20   
      10 /50  Test #10 : GGeoTest.GMaterialLibTest                     ***Exception: Child aborted    0.21   
      13 /50  Test #13 : GGeoTest.GScintillatorLibTest                 ***Exception: SegFault         0.21   
      16 /50  Test #16 : GGeoTest.GBndLibTest                          ***Exception: SegFault         0.20   
      17 /50  Test #17 : GGeoTest.GBndLibInitTest                      ***Exception: Child aborted    0.21   
      30 /50  Test #30 : GGeoTest.GPmtTest                             ***Exception: Child aborted    0.19   
      31 /50  Test #31 : GGeoTest.BoundariesNPYTest                    ***Exception: Child aborted    0.21   
      32 /50  Test #32 : GGeoTest.GAttrSeqTest                         ***Exception: Child aborted    0.21   
      36 /50  Test #36 : GGeoTest.GGeoLibTest                          ***Exception: Child aborted    0.19   
      37 /50  Test #37 : GGeoTest.GGeoTest                             ***Exception: Child aborted    0.17   
      38 /50  Test #38 : GGeoTest.GMakerTest                           ***Exception: Child aborted    0.20   
      45 /50  Test #45 : GGeoTest.GSurfaceLibTest                      ***Exception: Child aborted    0.19   
      47 /50  Test #47 : GGeoTest.NLookupTest                          ***Exception: Child aborted    0.17   
      48 /50  Test #48 : GGeoTest.RecordsNPYTest                       ***Exception: SegFault         0.21   
      49 /50  Test #49 : GGeoTest.GSceneTest                           ***Exception: Child aborted    0.17   
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 ***Exception: Child aborted    0.21   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 ***Exception: Child aborted    0.21   
      9  /18  Test #9  : OptiXRapTest.OOboundaryTest                   ***Exception: Child aborted    0.15   
      10 /18  Test #10 : OptiXRapTest.OOboundaryLookupTest             ***Exception: Child aborted    0.15   
      14 /18  Test #14 : OptiXRapTest.OEventTest                       ***Exception: Child aborted    0.24   
      15 /18  Test #15 : OptiXRapTest.OInterpolationTest               ***Exception: Child aborted    0.23   
      16 /18  Test #16 : OptiXRapTest.ORayleighTest                    ***Exception: Child aborted    0.17   
      1  /5   Test #1  : OKOPTest.OpIndexerTest                        ***Exception: Child aborted    0.16   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         ***Exception: Child aborted    0.16   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           ***Exception: Child aborted    0.13   
      1  /1   Test #1  : OpticksGLTest.OOAxisAppCheck                  ***Exception: Child aborted    0.30   
      2  /5   Test #2  : OKTest.OKTest                                 ***Exception: Child aborted    0.31   
      3  /5   Test #3  : OKTest.OTracerTest                            ***Exception: Child aborted    0.32   
      1  /24  Test #1  : CFG4Test.CMaterialLibTest                     ***Exception: SegFault         0.31   
      2  /24  Test #2  : CFG4Test.CMaterialTest                        ***Exception: SegFault         0.30   
      3  /24  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         0.30   
      5  /24  Test #5  : CFG4Test.CGDMLDetectorTest                    ***Exception: SegFault         0.32   
      6  /24  Test #6  : CFG4Test.CGeometryTest                        ***Exception: Child aborted    0.27   
      7  /24  Test #7  : CFG4Test.CG4Test                              ***Exception: SegFault         0.31   
      18 /24  Test #18 : CFG4Test.CCollectorTest                       ***Exception: Child aborted    0.25   
      19 /24  Test #19 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         0.31   
      21 /24  Test #21 : CFG4Test.CGROUPVELTest                        ***Exception: SegFault         0.26   
      24 /24  Test #24 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    0.29   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    0.51   



