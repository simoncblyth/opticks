after_avoid_templated_track_info_get_test_fails
=================================================


Fixed after re-export geometry::

    SLOW: tests taking longer that 15.0 seconds

    FAILS:  0   / 221   :  Tue Nov 25 18:52:12 2025  :  GEOM J25_4_0_opticks_Debug  



Looks like missing geometry::


    FAILS:  26  / 221   :  Tue Nov 25 17:32:34 2025  :  GEOM J25_4_0_opticks_Debug  
      1  /43  Test #1  : CSGTest.CSGNodeTest                                     ***Failed                      0.10   
      5  /43  Test #5  : CSGTest.CSGPrimSpecTest                                 ***Failed                      0.11   
      6  /43  Test #6  : CSGTest.CSGPrimTest                                     ***Failed                      0.11   
      8  /43  Test #8  : CSGTest.CSGFoundryTest                                  ***Failed                      0.11   
      10 /43  Test #10 : CSGTest.CSGFoundry_getCenterExtent_Test                 ***Failed                      0.10   
      11 /43  Test #11 : CSGTest.CSGFoundry_findSolidIdx_Test                    ***Failed                      0.11   
      14 /43  Test #14 : CSGTest.CSGNameTest                                     ***Failed                      0.10   
      15 /43  Test #15 : CSGTest.CSGTargetTest                                   ***Failed                      0.10   
      16 /43  Test #16 : CSGTest.CSGTargetGlobalTest                             ***Failed                      0.08   
      17 /43  Test #17 : CSGTest.CSGFoundry_MakeCenterExtentGensteps_Test        ***Failed                      0.18   
      18 /43  Test #18 : CSGTest.CSGFoundry_getFrame_Test                        ***Failed                      0.10   
      19 /43  Test #19 : CSGTest.CSGFoundry_getFrameE_Test                       ***Failed                      0.10   
      20 /43  Test #20 : CSGTest.CSGFoundry_getMeshName_Test                     ***Failed                      0.11   
      23 /43  Test #23 : CSGTest.CSGFoundryLoadTest                              ***Failed                      0.11   
      24 /43  Test #24 : CSGTest.CSGScanTest                                     ***Failed                      0.11   
      28 /43  Test #28 : CSGTest.CSGQueryTest                                    ***Failed                      0.10   
      29 /43  Test #29 : CSGTest.CSGSimtraceTest                                 ***Failed                      0.10   
      30 /43  Test #30 : CSGTest.CSGSimtraceRerunTest                            ***Failed                      0.10   
      31 /43  Test #31 : CSGTest.CSGSimtraceSampleTest                           ***Failed                      0.10   
      32 /43  Test #32 : CSGTest.CSGCopyTest                                     ***Failed                      0.09   
      18 /22  Test #18 : QUDARapTest.QOpticalTest                                ***Failed                      0.01   
      19 /22  Test #19 : QUDARapTest.QSimWithEventTest                           ***Failed                      0.37   
      20 /22  Test #20 : QUDARapTest.QSimTest                                    ***Failed                      0.37   
      3  /4   Test #3  : CSGOptiXTest.CSGOptiXRenderTest                         ***Failed                      0.11   
      1  /2   Test #1  : G4CXTest.G4CXRenderTest                                 ***Failed                      0.41   
      2  /2   Test #2  : G4CXTest.G4CXOpticks_setGeometry_Test                   ***Failed                      0.14   




