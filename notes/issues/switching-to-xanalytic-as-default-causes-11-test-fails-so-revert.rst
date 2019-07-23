switching-to-xanalytic-as-default-causes-11-test-fails-so-revert
======================================================================



::

    FAILS:  11  / 409   :  Tue Jul 23 22:21:31 2019   
      4  /24  Test #4  : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     2.01   
      12 /24  Test #12 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     1.40   
      18 /24  Test #18 : OptiXRapTest.interpolationTest                Child aborted***Exception:     1.44   
      21 /24  Test #21 : OptiXRapTest.intersectAnalyticTest.iaTorusTest Child aborted***Exception:     2.27   
      1  /5   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     1.33   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     1.85   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     1.42   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     1.74   
      3  /5   Test #3  : OKTest.OTracerTest                            Child aborted***Exception:     1.74   
      30 /34  Test #30 : CFG4Test.CAlignEngineTest                     Child aborted***Exception:     0.26   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     2.88   
    [blyth@localhost opticks]$ 



Tripping the somePrim assert::


    2019-07-23 22:33:03.683 INFO  [11049] [OGeo::convert@219] [ nmm 6
    2019-07-23 22:33:03.684 FATAL [11049] [OGeo::makeAnalyticGeometry@667]  NodeTree : MISMATCH (numPrim != numVolumes)  (this happens when using --csgskiplv)  numVolumes 12230 numVolumesSelected 0 numPrim 0 numPart 0 numTran 0 numPlan 0
    2019-07-23 22:33:03.684 FATAL [11049] [OGeo::makeAnalyticGeometry@701]  someprim fails  mm.index 0 numPrim 0 numPart 0 numTran 0 numPlan 0
    OKTest: /home/blyth/opticks/optixrap/OGeo.cc:710: optix::Geometry OGeo::makeAnalyticGeometry(GMergedMesh*, unsigned int): Assertion `someprim' failed.
    Aborted (core dumped)




