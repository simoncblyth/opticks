opticks_t_1_jun_2021_12_of_471_fails
========================================


::

    SLOW: tests taking longer that 15 seconds
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     15.22  


    FAILS:  12  / 471   :  Wed Jun  2 04:39:01 2021   
      19 /119 Test #19 : NPYTest.TorchStepNPYTest                      Child aborted***Exception:     0.07   
      32 /45  Test #32 : OpticksCoreTest.OpticksGenstepTest            Child aborted***Exception:     0.06   

          FIXED : was doubling the  number of steps, after mobe to NStep getOneStep() approach 


      42 /45  Test #42 : OpticksCoreTest.OpticksEventLeakTest          Child aborted***Exception:     0.07   
      43 /45  Test #43 : OpticksCoreTest.OpticksRunTest                Child aborted***Exception:     0.08   

          FIXED

      22 /33  Test #22 : OptiXRapTest.eventTest                        Child aborted***Exception:     4.44   
      23 /33  Test #23 : OptiXRapTest.interpolationTest                ***Failed                      5.10   
      2  /6   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     9.55   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     9.73   
      8  /41  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     9.19   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     15.22  
      1  /2   Test #1  : G4OKTest.G4OKTest                             Child aborted***Exception:     10.40  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.18   
    O[blyth@localhost opticks]$ 





    FAILS:  25  / 471   :  Wed Jun  2 05:21:29 2021   
      43 /45  Test #43 : OpticksCoreTest.OpticksRunTest                Child aborted***Exception:     0.09   

           FIXED

      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Child aborted***Exception:     2.22   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 Child aborted***Exception:     2.19   


      17 /33  Test #17 : OptiXRapTest.rayleighTest                     Child aborted***Exception:     2.30   
      22 /33  Test #22 : OptiXRapTest.eventTest                        Child aborted***Exception:     3.97   
      23 /33  Test #23 : OptiXRapTest.interpolationTest                Child aborted***Exception:     2.64   
      1  /6   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     2.30   
      2  /6   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     9.35   
      5  /6   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     2.64   
      6  /6   Test #6  : OKOPTest.OpFlightPathTest                     Child aborted***Exception:     3.39   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     2.77   
      3  /5   Test #3  : OKTest.OTracerTest                            Child aborted***Exception:     2.26   
      1  /41  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     2.46   
      2  /41  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     3.45   
      3  /41  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     2.42   
      5  /41  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     2.31   
      7  /41  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     2.26   
      8  /41  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     2.25   
      27 /41  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     2.32   
      29 /41  Test #29 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     2.26   
      33 /41  Test #33 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     2.24   
      36 /41  Test #36 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     2.30   
      37 /41  Test #37 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     2.29   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     2.30   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.10   
    O[blyth@localhost opticks]$ 
    O[blyth@localhost opticks]$ 

