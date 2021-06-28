opticks_t_44_of_486_after_switch_to_double_props
==================================================




okc/Opticks.cc bump Opticks::GEOCACHE_CODE_VERSION to 10 to force recreation of geocache
--------------------------------------------------------------------------------------------

::

     404 geocache-jun28-gdmlpath(){ echo $(opticks-prefix)/origin_CGDMLKludge_jun28.gdml ; }
     405 geocache-jun28(){
     406     local msg="=== $FUNCNAME :"
     407     local path=$(geocache-jun28-gdmlpath)
     408     # get skips from current tds3
     409     local skipsolidname="mask_PMT_20inch_vetosMask_virtual,NNVTMCPPMT_body_solid,HamamatsuR12860_body_solid_1_9,PMT_20inch_veto_body_solid_1_2"
     410     GTree=INFO OpticksDbg=INFO GInstancer=INFO geocache-create- --gdmlpath $path -D --noviz  --skipsolidname $skipsolidname $*  
     411 }   



Lots of errors from failed array loading due to expecting double
-------------------------------------------------------------------

::

    FAILS:  44  / 486   :  Mon Jun 28 07:39:16 2021   
      13 /58  Test #13 : GGeoTest.GScintillatorLibTest                 Child aborted***Exception:     0.09   
      16 /58  Test #16 : GGeoTest.GBndLibTest                          Child aborted***Exception:     0.08   
      17 /58  Test #17 : GGeoTest.GBndLibInitTest                      Child aborted***Exception:     0.08   
      31 /58  Test #31 : GGeoTest.GPtsTest                             Child aborted***Exception:     0.41   
      35 /58  Test #35 : GGeoTest.BoundariesNPYTest                    Child aborted***Exception:     0.08   
      40 /58  Test #40 : GGeoTest.GGeoLibTest                          Child aborted***Exception:     0.07   
      41 /58  Test #41 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.07   
      42 /58  Test #42 : GGeoTest.GGeoIdentityTest                     Child aborted***Exception:     0.08   
      43 /58  Test #43 : GGeoTest.GGeoConvertTest                      Child aborted***Exception:     0.07   
      45 /58  Test #45 : GGeoTest.GMakerTest                           Child aborted***Exception:     0.06   
      52 /58  Test #52 : GGeoTest.GSurfaceLibTest                      Child aborted***Exception:     0.06   
      54 /58  Test #54 : GGeoTest.RecordsNPYTest                       Child aborted***Exception:     0.08   
      57 /58  Test #57 : GGeoTest.GPhoTest                             Child aborted***Exception:     0.07   
      58 /58  Test #58 : GGeoTest.GGeoDumpTest                         Child aborted***Exception:     0.08   
      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Child aborted***Exception:     0.08   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 Child aborted***Exception:     0.08   
      3  /3   Test #3  : OpticksGeoTest.OpticksHubGGeoTest             Child aborted***Exception:     0.35   
      3  /35  Test #3  : OptiXRapTest.OScintillatorLibTest             Child aborted***Exception:     0.19   
      11 /35  Test #11 : OptiXRapTest.textureTest                      Child aborted***Exception:     0.18   
      12 /35  Test #12 : OptiXRapTest.boundaryTest                     Child aborted***Exception:     0.20   
      13 /35  Test #13 : OptiXRapTest.reemissionTest                   Child aborted***Exception:     0.20   
      15 /35  Test #15 : OptiXRapTest.boundaryLookupTest               Child aborted***Exception:     0.23   
      19 /35  Test #19 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     0.19   
      24 /35  Test #24 : OptiXRapTest.eventTest                        Child aborted***Exception:     0.17   
      25 /35  Test #25 : OptiXRapTest.interpolationTest                Child aborted***Exception:     0.19   
      1  /6   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     0.19   
      2  /6   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     0.19   
      5  /6   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     0.19   
      6  /6   Test #6  : OKOPTest.OpFlightPathTest                     Child aborted***Exception:     0.19   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     0.20   
      3  /5   Test #3  : OKTest.OTracerTest                            Child aborted***Exception:     0.21   
      1  /46  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     0.75   
      2  /46  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     0.25   
      3  /46  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     0.26   
      5  /46  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     0.24   
      7  /46  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     0.22   
      8  /46  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     0.24   
      28 /46  Test #28 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     0.40   
      30 /46  Test #30 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     0.23   
      38 /46  Test #38 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     0.23   
      39 /46  Test #39 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     0.24   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     0.28   
      1  /2   Test #1  : G4OKTest.G4OKTest                             Child aborted***Exception:     0.25   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      3.25   
    O[blyth@localhost opticks]$ 

