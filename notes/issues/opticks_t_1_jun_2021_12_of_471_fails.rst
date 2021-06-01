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
      22 /33  Test #22 : OptiXRapTest.eventTest                        Child aborted***Exception:     4.44   
      23 /33  Test #23 : OptiXRapTest.interpolationTest                ***Failed                      5.10   
      2  /6   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     9.55   
      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     9.73   
      8  /41  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     9.19   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     15.22  
      1  /2   Test #1  : G4OKTest.G4OKTest                             Child aborted***Exception:     10.40  
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.18   
    O[blyth@localhost opticks]$ 

