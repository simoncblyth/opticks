opticks_t_1_jun_2021_12_of_471_fails
========================================



::

    O[blyth@localhost cfg4]$ gdb OKG4Test 

    2021-06-02 06:25:32.618 INFO  [65599] [OGeo::convert@321] ] nmm 10
    2021-06-02 06:25:32.689 ERROR [65599] [cuRANDWrapper::setItems@154] CAUTION : are resizing the launch sequence 
    2021-06-02 06:25:33.635 INFO  [65599] [OpticksRun::createOKEvent@158]  tagoffset 0 skipaheadstep 0 skipahead 0
    2021-06-02 06:25:33.640 INFO  [65599] [CG4::propagate@411]  calling BeamOn numG4Evt 1
    2021-06-02 06:27:44.049 INFO  [65599] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 5 : BAD_FLAG
    OKG4Test: /home/blyth/opticks/cfg4/CWriter.cc:310: void CWriter::writeStepPoint_(const G4StepPoint*, const CPhoton&): Assertion `m_target_records' failed.

    (gdb) bt
    #3  0x00007fffe5743252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff4adf998 in CWriter::writeStepPoint_ (this=0x90da0e0, point=0x9068870, photon=...) at /home/blyth/opticks/cfg4/CWriter.cc:310
    #5  0x00007ffff4adf801 in CWriter::writeStepPoint (this=0x90da0e0, point=0x9068870, flag=4096, material=1, last=false) at /home/blyth/opticks/cfg4/CWriter.cc:263
    #6  0x00007ffff4ad62dc in CRecorder::WriteStepPoint (this=0x1bed6680, point=0x9068870, flag=4096, material=1, boundary_status=Undefined, last=false) at /home/blyth/opticks/cfg4/CRecorder.cc:713
    #7  0x00007ffff4ad5af4 in CRecorder::postTrackWriteSteps (this=0x1bed6680) at /home/blyth/opticks/cfg4/CRecorder.cc:615
    #8  0x00007ffff4ad3faf in CRecorder::postTrack (this=0x1bed6680) at /home/blyth/opticks/cfg4/CRecorder.cc:230
    #9  0x00007ffff4aff064 in CManager::postTrack (this=0x1be71840) at /home/blyth/opticks/cfg4/CManager.cc:314
    #10 0x00007ffff4afefd8 in CManager::PostUserTrackingAction (this=0x1be71840, track=0x3ea9b140) at /home/blyth/opticks/cfg4/CManager.cc:296
    #11 0x00007ffff4af7efb in CTrackingAction::PostUserTrackingAction (this=0x8ee0b70, track=0x3ea9b140) at /home/blyth/opticks/cfg4/CTrackingAction.cc:79
    #12 0x00007ffff190d14d in G4TrackingManager::ProcessOneTrack(G4Track*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #13 0x00007ffff1b44b53 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #14 0x00007ffff1de1b27 in G4RunManager::ProcessOneEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #15 0x00007ffff1ddabd3 in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #16 0x00007ffff1dda99e in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #17 0x00007ffff4afca75 in CG4::propagate (this=0x8ed3490) at /home/blyth/opticks/cfg4/CG4.cc:414
    #18 0x00007ffff7bafde4 in OKG4Mgr::propagate_ (this=0x7fffffffad20) at /home/blyth/opticks/okg4/OKG4Mgr.cc:220
    #19 0x00007ffff7bafc6e in OKG4Mgr::propagate (this=0x7fffffffad20) at /home/blyth/opticks/okg4/OKG4Mgr.cc:158




::

    FAILS:  3   / 471   :  Wed Jun  2 06:39:01 2021   
      8  /41  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     9.39   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     148.35 
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      7.01   
    O[blyth@localhost opticks]$ 


::

    FAILS:  4   / 471   :  Wed Jun  2 06:18:34 2021   
      2  /6   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     9.16   
      8  /41  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     9.39   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     149.14 
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.95   
    O[blyth@localhost opticks]$ 






::

    SLOW: tests taking longer that 15 seconds
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     146.26 


    FAILS:  5   / 471   :  Wed Jun  2 05:59:32 2021   
      22 /33  Test #22 : OptiXRapTest.eventTest                        Child aborted***Exception:     3.83   

          FIXED 

      2  /6   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     9.31   

          FIXED ctrl bool again

      8  /41  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     9.15   

           

      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     146.26 
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.11   
    O[blyth@localhost opticks]$ 




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

          FIXED has old boolean ctrl, not the new char 

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

          FIXED WAS LACK OF SETTING TARGET  

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

