opticks_t_1_jun_2021_12_of_471_fails
========================================



Jun 4 
-------


::


    FAILS:  3   / 474   :  Fri Jun  4 23:42:27 2021   
      8  /43  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     9.32   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     15.38  

           CManager m_mode asserts : FIXED 


      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      6.19   
    O[blyth@localhost opticks]$ 


Move one remaining fail to: 

* :doc:`tboolean_box_fail`



Jun 4 
-------

::

    FAILS:  7   / 474   :  Fri Jun  4 23:00:50 2021   
      3  /43  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     9.30   
      8  /43  Test #8  : CFG4Test.CG4Test                              Child aborted***Exception:     9.27   
      28 /43  Test #28 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     9.32   
      33 /43  Test #33 : CFG4Test.CPhotonTest                          Child aborted***Exception:     0.22   
      34 /43  Test #34 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     9.38   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     9.29   

            CCtx ctor is too soon for CGenstepCollector::Get() causing assert


      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      4.89   
    O[blyth@localhost cfg4]$ 



CTestDetectorTest/... : CCtx ctor is too soon for CGenstepCollector::Get() causing assert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    (gdb) bt
    #0  0x00007fffe877f387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe8780a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe87781a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe8778252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b3f4e4 in CGenstepCollector::Get () at /home/blyth/opticks/cfg4/CGenstepCollector.cc:46
    #5  0x00007ffff7b3675f in CCtx::CCtx (this=0x1bea9020, ok=0x7fffffffa750) at /home/blyth/opticks/cfg4/CCtx.cc:54
    #6  0x00007ffff7b3bf86 in CManager::CManager (this=0x1be31810, ok=0x7fffffffa750) at /home/blyth/opticks/cfg4/CManager.cc:80
    #7  0x00007ffff7b3942d in CG4::CG4 (this=0x7fffffffa670, hub=0x7fffffffa5e0) at /home/blyth/opticks/cfg4/CG4.cc:171
    #8  0x0000000000403c09 in main (argc=1, argv=0x7fffffffaf88) at /home/blyth/opticks/cfg4/tests/CTestDetectorTest.cc:52
    (gdb) 





::

    gdb --args OKG4Test --managermode 3 



This one is lack of calling CManager::BeginOfGenstep from S+C following move to genstep chunking
---------------------------------------------------------------------------------------------------

* also CGenerator was using old dynamic=false for all sources other than gun
* switched to new onestep approach

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

    13PMT_20inch_veto_photocathode_logsurf2                Photocathode_opsurf pv1 PMT_20inch_veto_body_phys0x3c3e550 #0 pv2 PMT_20inch_veto_inner1_phys0x3c3e5d0 #0
       14                     CDTyvekSurface              CDTyvekOpticalSurface pv1 pOuterWaterPool0x3491360 #0 pv2 pCentralDetector0x3493130 #0
    2021-06-02 06:42:40.321 WARN  [104477] [main@50]  post CG4 
    2021-06-02 06:42:40.321 WARN  [104477] [main@54]   post CG4::interactive
    2021-06-02 06:42:40.321 ERROR [104477] [main@59]  setting gensteps 0x8e92af0
    2021-06-02 06:42:40.322 INFO  [104477] [OpticksRun::createOKEvent@158]  tagoffset 0 skipaheadstep 0 skipahead 0
    2021-06-02 06:42:40.324 INFO  [104477] [CG4::propagate@411]  calling BeamOn numG4Evt 1
    2021-06-02 06:44:39.018 INFO  [104477] [CTorchSource::GeneratePrimaryVertex@293]  event_gencode 5 : BAD_FLAG
    CG4Test: /home/blyth/opticks/cfg4/CWriter.cc:310: void CWriter::writeStepPoint_(const G4StepPoint*, const CPhoton&): Assertion `m_target_records' failed.

    (gdb) bt
    #3  0x00007fffe8787252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff7b23998 in CWriter::writeStepPoint_ (this=0x9082e60, point=0x9028790, photon=...) at /home/blyth/opticks/cfg4/CWriter.cc:310
    #5  0x00007ffff7b23801 in CWriter::writeStepPoint (this=0x9082e60, point=0x9028790, flag=4096, material=1, last=false) at /home/blyth/opticks/cfg4/CWriter.cc:263
    #6  0x00007ffff7b1a2dc in CRecorder::WriteStepPoint (this=0x9099b70, point=0x9028790, flag=4096, material=1, boundary_status=Undefined, last=false) at /home/blyth/opticks/cfg4/CRecorder.cc:713
    #7  0x00007ffff7b19af4 in CRecorder::postTrackWriteSteps (this=0x9099b70) at /home/blyth/opticks/cfg4/CRecorder.cc:615
    #8  0x00007ffff7b17faf in CRecorder::postTrack (this=0x9099b70) at /home/blyth/opticks/cfg4/CRecorder.cc:230
    #9  0x00007ffff7b43064 in CManager::postTrack (this=0x1be31800) at /home/blyth/opticks/cfg4/CManager.cc:314
    #10 0x00007ffff7b42fd8 in CManager::PostUserTrackingAction (this=0x1be31800, track=0x23fa8640) at /home/blyth/opticks/cfg4/CManager.cc:296
    #11 0x00007ffff7b3befb in CTrackingAction::PostUserTrackingAction (this=0x8ea0a90, track=0x23fa8640) at /home/blyth/opticks/cfg4/CTrackingAction.cc:79
    #12 0x00007ffff495114d in G4TrackingManager::ProcessOneTrack(G4Track*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4tracking.so
    #13 0x00007ffff4b88b53 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #14 0x00007ffff4e25b27 in G4RunManager::ProcessOneEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #15 0x00007ffff4e1ebd3 in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #16 0x00007ffff4e1e99e in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #17 0x00007ffff7b40a75 in CG4::propagate (this=0x8e93090) at /home/blyth/opticks/cfg4/CG4.cc:414
    #18 0x000000000040427e in main (argc=1, argv=0x7fffffffade8) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:68





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

