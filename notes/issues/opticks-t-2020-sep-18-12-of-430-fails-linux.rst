opticks-t-2020-sep-18-12-of-430-fails-linux
==================================================


Now : no slow, 1 FAIL : IntegrationTests.tboolean.box 
-------------------------------------------------------

* :doc:`GGeoTest_GMergedMesh_mergeVolumeFaces_assert_sensor_indices`

::

    SLOW: tests taking longer that 15 seconds

    FAILS:  1   / 434   :  Fri Oct  2 00:42:36 2020   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      1.15   
    [blyth@localhost opticks]$ 



Now 2 FAIL and no slow
-------------------------

* slow OpSnapTest transient, flakiness suspicion remains (maybe globalinstance related)

::


    SLOW: tests taking longer that 15 seconds


    FAILS:  2   / 434   :  Thu Oct  1 22:32:49 2020   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         5.87   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      1.14   



* :doc:`G4NuclideTable_ENSDFSTATE_dat_is_not_found.rst`



Now 4
--------

Following old TORCH enum fix down to:  1 very slow and 4 fails. Using geocache-dx DYB geometry::

    SLOW: tests taking longer that 15 seconds
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Passed                         126.98 


    FAILS:  4   / 434   :  Thu Oct  1 00:52:26 2020   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         5.83   

           G4 env detection not working ?

      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     8.50   
      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     11.27  

           new genstep codes do not conform to (0x1 << n) expectation in CPhoton::add

      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      1.18   
    [blyth@localhost opticks]$ 



CTestDetectorTest
~~~~~~~~~~~~~~~~~~~

CTestDetectorTest, G4 environment detection fails ?::

    2020-10-01 01:35:26.604 INFO  [326824] [OpticksGen::targetGenstep@336] setting frame 0 Id
    2020-10-01 01:35:26.604 ERROR [326824] [OpticksGen::makeTorchstep@431]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-10-01 01:35:26.604 INFO  [326824] [BOpticksResource::IsGeant4EnvironmentDetected@287]  n 0 detect 0
    2020-10-01 01:35:26.604 ERROR [326824] [CG4::preinit@140] No external Geant4 environment, will setup internally using g4- config ini file 
    2020-10-01 01:35:26.604 ERROR [326824] [OpticksResource::SetupG4Environment@519] inipath /home/blyth/local/opticks/externals/config/geant4.ini

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : PART70001
          issued by : G4NuclideTable
    ENSDFSTATE.dat is not found.
    *** Fatal Exception *** core dump ***
    Segmentation fault (core dumped)

    [blyth@localhost opticks]$ env | grep G4
    G4LEVELGAMMADATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/PhotonEvaporation5.2
    G4NEUTRONXSDATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/G4NEUTRONXS1.4
    G4LEDATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/G4EMLOW7.3
    G4NEUTRONHPDATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/G4NDL4.5
    G4ENSDFSTATEDATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2
    G4RADIOACTIVEDATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/RadioactiveDecay5.2
    G4ABLADATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/G4ABLA3.1
    G4PIIDATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/G4PII1.3
    G4SAIDXSDATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/G4SAIDDATA1.1
    G4REALSURFACEDATA=/home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/share/Geant4-10.4.2/data/RealSurface2.1.1
    [blyth@localhost opticks]$ 




FIXED : CG4Test + OKG4Test, old TORCH enum again
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2020-10-01 01:38:18.582 WARN  [331071] [main@55]   post CG4::interactive
    2020-10-01 01:38:18.583 ERROR [331071] [main@63]  setting gensteps 0xa02afc0
    2020-10-01 01:38:18.585 INFO  [331071] [CG4::propagate@395]  calling BeamOn numG4Evt 1
    2020-10-01 01:38:20.500 INFO  [331071] [CTorchSource::GeneratePrimaryVertex@288]  event_gencode 4096 : TORCH
    CG4Test: /home/blyth/opticks/cfg4/CG4Ctx.cc:261: void CG4Ctx::setEvent(const G4Event*): Assertion `valid' failed.

    #4  0x00007ffff7b46d37 in CG4Ctx::setEvent (this=0xa02b6e0, event=0xc1a4640) at /home/blyth/opticks/cfg4/CG4Ctx.cc:261
    #5  0x00007ffff7b42ecb in CEventAction::setEvent (this=0xce82f30, event=0xc1a4640) at /home/blyth/opticks/cfg4/CEventAction.cc:69
    #6  0x00007ffff7b42e72 in CEventAction::BeginOfEventAction (this=0xce82f30, anEvent=0xc1a4640) at /home/blyth/opticks/cfg4/CEventAction.cc:59
    #7  0x00007ffff4b99875 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #8  0x00007ffff4e36b27 in G4RunManager::ProcessOneEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #9  0x00007ffff4e2fbd3 in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #10 0x00007ffff4e2f99e in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #11 0x00007ffff7b4a6fc in CG4::propagate (this=0xa02b6b0) at /home/blyth/opticks/cfg4/CG4.cc:398
    #12 0x00000000004048ca in main (argc=1, argv=0x7fffffff7398) at /home/blyth/opticks/cfg4/tests/CG4Test.cc:71
    (gdb) f 4
    #4  0x00007ffff7b46d37 in CG4Ctx::setEvent (this=0xa02b6e0, event=0xc1a4640) at /home/blyth/opticks/cfg4/CG4Ctx.cc:261
    261	    assert( valid );


    (gdb) list
    246	    CEventInfo* eui = (CEventInfo*)event->GetUserInformation(); 
    247	    assert(eui && "expecting event UserInfo set by eg CGenstepSource "); 
    248	
    249	    _gen = eui->gencode ;
    250	
    251	
    252	    bool valid = OpticksGenstep::IsValid(_gen) ; 
    253	
    254	    LOG(LEVEL) 
    255	        << " gen " << _gen
    (gdb) p _gen
    $1 = 4096



    2020-10-01 01:42:28.990 FATAL [337256] [SLog::operator@47]  ) OPropagator::OPropagator  DONE
    2020-10-01 01:42:28.995 INFO  [337256] [CG4::propagate@395]  calling BeamOn numG4Evt 1
    2020-10-01 01:42:30.974 INFO  [337256] [CTorchSource::GeneratePrimaryVertex@288]  event_gencode 4096 : TORCH
    OKG4Test: /home/blyth/opticks/cfg4/CG4Ctx.cc:261: void CG4Ctx::setEvent(const G4Event*): Assertion `valid' failed.

    (gdb) bt
    #4  0x00007ffff4c4cd37 in CG4Ctx::setEvent (this=0xa06a200, event=0x13584f20) at /home/blyth/opticks/cfg4/CG4Ctx.cc:261
    #5  0x00007ffff4c48ecb in CEventAction::setEvent (this=0xcebaa20, event=0x13584f20) at /home/blyth/opticks/cfg4/CEventAction.cc:69
    #6  0x00007ffff4c48e72 in CEventAction::BeginOfEventAction (this=0xcebaa20, anEvent=0x13584f20) at /home/blyth/opticks/cfg4/CEventAction.cc:59
    #7  0x00007ffff1c9f875 in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #8  0x00007ffff1f3cb27 in G4RunManager::ProcessOneEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #9  0x00007ffff1f35bd3 in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #10 0x00007ffff1f3599e in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4run.so
    #11 0x00007ffff4c506fc in CG4::propagate (this=0xa06a1d0) at /home/blyth/opticks/cfg4/CG4.cc:398
    #12 0x00007ffff7bd4b7f in OKG4Mgr::propagate_ (this=0x7fffffff7050) at /home/blyth/opticks/okg4/OKG4Mgr.cc:220
    #13 0x00007ffff7bd4a1a in OKG4Mgr::propagate (this=0x7fffffff7050) at /home/blyth/opticks/okg4/OKG4Mgr.cc:158
    #14 0x0000000000403a99 in main (argc=1, argv=0x7fffffff7398) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:28
    (gdb) 



CG4Test + OKG4Test, CPhoton::add expecting flag (0x1 << n)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

After fixing the above, get another. The history nibble recording assumes flags are (0x1 << n) where n in 0..15. 
That is no longer the case for the first genstep flag. So need to translate from OpticksGenstep enum code to OpticksPhoton 
code. Added OpticksGenstep::GenstepToPhotonFlag for this::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff7c6edb66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff7c8b8080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff7c6491ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7c6111ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001ced29 libCFG4.dylib`CPhoton::add(this=0x000000011a8e08a8, flag=5, material=14) at CPhoton.cc:130
        frame #5: 0x00000001001d18a7 libCFG4.dylib`CWriter::writeStepPoint(this=0x000000011a8e0a10, point=0x000000011eb97ac0, flag=5, material=14, last=false) at CWriter.cc:172
        frame #6: 0x00000001001bfac2 libCFG4.dylib`CRecorder::WriteStepPoint(this=0x000000011a8e0860, point=0x000000011eb97ac0, flag=5, material=14, boundary_status=Undefined, (null)="PRE", last=false) at CRecorder.cc:613
      * frame #7: 0x00000001001bed86 libCFG4.dylib`CRecorder::postTrackWriteSteps(this=0x000000011a8e0860) at CRecorder.cc:529
        frame #8: 0x00000001001bdb38 libCFG4.dylib`CRecorder::postTrack(this=0x000000011a8e0860) at CRecorder.cc:179
        frame #9: 0x00000001001fd251 libCFG4.dylib`CG4::postTrack(this=0x000000011a6a8560) at CG4.cc:320
        frame #10: 0x00000001001f6fbe libCFG4.dylib`CTrackingAction::PostUserTrackingAction(this=0x000000011a8e0b90, track=0x000000011eb96d90) at CTrackingAction.cc:114
        frame #11: 0x00000001020f9937 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x00000001191bf760, apValueG4Track=0x000000011eb96d90) at G4TrackingManager.cc:140
        frame #12: 0x0000000101fbf71a libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001191bf6d0, anEvent=0x000000011e142d30) at G4EventManager.cc:185
        frame #13: 0x0000000101fc0c2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x00000001191bf6d0, anEvent=0x000000011e142d30) at G4EventManager.cc:338
        frame #14: 0x0000000101ecc9f5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x00000001165633a0, i_event=0) at G4RunManager.cc:399
        frame #15: 0x0000000101ecc825 libG4run.dylib`G4RunManager::DoEventLoop(this=0x00000001165633a0, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #16: 0x0000000101ecace1 libG4run.dylib`G4RunManager::BeamOn(this=0x00000001165633a0, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #17: 0x00000001001fe134 libCFG4.dylib`CG4::propagate(this=0x000000011a6a8560) at CG4.cc:398
        frame #18: 0x00000001000107e0 CG4Test`main(argc=1, argv=0x00007ffeefbfea00) at CG4Test.cc:71
        frame #19: 0x00007fff7c59d015 libdyld.dylib`start + 1
    (lldb) 


    404 void CRecorder::postTrackWriteSteps()
    ...
    522         unsigned preFlag = first ? m_ctx._gen : OpStatus::OpPointFlag(pre,  prior_boundary_status, stage) ;
    523 
    524         if(i == 0)
    525         {
    526 
    527             m_state._step_action |= CAction::PRE_SAVE ;
    528 
    529             done = WriteStepPoint( pre , preFlag,  u_premat,  prior_boundary_status, PRE, false);
    530 

    /// CPhoton
    111 
    112     _his = SBit::ffs(flag) & 0xFull ;
    113 
    114     //  SBit::ffs result is a 1-based bit index of least significant set bit 
    115     //  so anding with 0xF although looking like a bug, as the result of ffs is not a nibble, 
    116     //  is actually providing a warning as are constructing seqhis from nibbles : 
    117     //  this is showing that NATURAL is too big to fit in its nibble   
    118     //
    119     //  BUT NATURAL is an input flag meaning either CERENKOV or SCINTILATION, thus
    120     //  it should not be here at the level of a photon.  It needs to be set 
    121     //  at genstep level to the appropriate thing. 
    122     //
    123     //  See notes/issues/ckm-okg4-CPhoton-add-flag-mismatch-NATURAL-bit-index-too-big-for-nibble.rst      
    124     //
    125 
    126     _flag = 0x1 << (_his - 1) ;
    127 
    128     bool flag_match = _flag == flag  ;
    129     if(!flag_match)
    130        LOG(fatal) << "flag mismatch "
    131                   << " (expecting [0x1 << 0..15]) "
    132                   << " TOO BIG TO FIT IN THE NIBBLE "
    133                   << " _his " << _his
    134                   << " flag(input) " << flag
    135                   << " _flag(recon) " << _flag
    136                   ;
    137      assert( flag_match );



::

    349 void CG4::initEvent(OpticksEvent* evt)
    350 {
    351     LOG(LEVEL) << "[" ;
    352     m_generator->configureEvent(evt);
    353 
    354     m_ctx.initEvent(evt);
    355 
    356     m_recorder->initEvent(evt);
    357 
    358     NPY<float>* nopstep = evt->getNopstepData();
    359     if(!nopstep) LOG(fatal) << " nopstep NULL " << " evt " << evt->getShapeString() ;
    360     assert(nopstep);
    361     m_steprec->initEvent(nopstep);
    362     LOG(LEVEL) << "]" ;
    363 }







Adding OPTICKS_PYTHON to pick the python with numpy reduces fails from 12 to 10::


    FAILS:  10  / 430   :  Sat Sep 26 23:03:56 2020   
      30 /53  Test #30 : GGeoTest.GPtsTest                             ***Failed                      0.37   

            cannot compare : suspect deferred GParts as standard makes this test useless 
            for now switch off the fail, and see if this is correct

      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     10.04  

        2020-09-26 23:11:19.777 ERROR [146867] [G4StepNPY::checkGencodes@272]  i 0 unexpected label 4096
        2020-09-26 23:11:19.777 FATAL [146867] [G4StepNPY::checkGencodes@283] G4StepNPY::checklabel FAIL numStep 1 mismatch 1
        OKTest: /home/blyth/opticks/npy/G4StepNPY.cpp:288: void G4StepNPY::checkGencodes(): Assertion `mismatch == 0' failed.

        2020-09-26 23:26:26.079 ERROR [172407] [G4StepNPY::checkGencodes@281]  i 0 unexpected gencode label 4096 allowed gencodes 5,
        2020-09-26 23:26:26.079 FATAL [172407] [G4StepNPY::checkGencodes@293] G4StepNPY::checklabel FAIL numStep 1 mismatch 1
        OKTest: /home/blyth/opticks/npy/G4StepNPY.cpp:298: void G4StepNPY::checkGencodes(): Assertion `mismatch == 0' failed.



        #3  0x00007fffeacb40d2 in __assert_fail () from /lib64/libc.so.6
        #4  0x00007ffff29645a4 in G4StepNPY::checkGencodes (this=0x225ad8c0) at /home/blyth/opticks/npy/G4StepNPY.cpp:288
        #5  0x00007ffff2e7b1bf in OpticksRun::importGenstepData (this=0x678a60, gs=0x57684e0, oac_label=0x0) at /home/blyth/opticks/optickscore/OpticksRun.cc:423
        #6  0x00007ffff2e7a396 in OpticksRun::importGensteps (this=0x678a60) at /home/blyth/opticks/optickscore/OpticksRun.cc:253
        #7  0x00007ffff2e7a290 in OpticksRun::setGensteps (this=0x678a60, gensteps=0x57684e0) at /home/blyth/opticks/optickscore/OpticksRun.cc:225
        #8  0x00007ffff7bd524e in OKMgr::propagate (this=0x7fffffffad70) at /home/blyth/opticks/ok/OKMgr.cc:123
        #9  0x0000000000402f0c in main (argc=1, argv=0x7fffffffaee8) at /home/blyth/opticks/ok/tests/OKTest.cc:32
        (gdb) 
        (gdb) f 8
        #8  0x00007ffff7bd524e in OKMgr::propagate (this=0x7fffffffad70) at /home/blyth/opticks/ok/OKMgr.cc:123
        123             m_run->setGensteps(m_gen->getInputGensteps()); 
        (gdb) f 7
        #7  0x00007ffff2e7a290 in OpticksRun::setGensteps (this=0x678a60, gensteps=0x57684e0) at /home/blyth/opticks/optickscore/OpticksRun.cc:225
        225     importGensteps();
        (gdb) f 6
        #6  0x00007ffff2e7a396 in OpticksRun::importGensteps (this=0x678a60) at /home/blyth/opticks/optickscore/OpticksRun.cc:253
        253     m_g4step = importGenstepData(m_gensteps, oac_label) ;
        (gdb) p m_gensteps
        $1 = (NPY<float> *) 0x57684e0
        (gdb) p m_gensteps->getShapeString()
        Too few arguments in function call.
        (gdb) p m_gensteps->getShapeString(0)
        $2 = "1,6,4"
        (gdb) 

        (gdb) f 5
        #5  0x00007ffff2e7b1bf in OpticksRun::importGenstepData (this=0x678a60, gs=0x57684e0, oac_label=0x0) at /home/blyth/opticks/optickscore/OpticksRun.cc:423
        423     g4step->checkGencodes();
        (gdb) f 4
        #4  0x00007ffff29645a4 in G4StepNPY::checkGencodes (this=0x225ad8c0) at /home/blyth/opticks/npy/G4StepNPY.cpp:288
        288     assert(mismatch == 0 );
        (gdb) l
        283          LOG(fatal)<<"G4StepNPY::checklabel FAIL" 
        284                    << " numStep " << numStep
        285                    << " mismatch " << mismatch ; 
        286                    ;
        287     }
        288     assert(mismatch == 0 );
        289 }
        290 

        Probably old gensteps not adhering to the new enum codes   


        blyth@localhost optickscore]$ OpticksGenstepTest 
        2020-09-26 23:39:37.477 INFO  [196742] [main@32] OpticksGenstep::Dump()
        2020-09-26 23:39:37.478 INFO  [196742] [main@33] 
                 0 : INVALID
                 1 : G4Cerenkov_1042
                 2 : G4Scintillation_1042
                 3 : DsG4Cerenkov_r3971
                 4 : DsG4Scintillation_r3971
                 5 : torch
                 6 : fabricated
                 7 : emitsource
                 8 : natural
                 9 : machinery
                10 : g4gun
                11 : primarysource
                12 : genstepsource






      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         1.09   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     1.04   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     1.06   
      7  /34  Test #7  : CFG4Test.CG4Test                              ***Exception: SegFault         1.13   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         1.16   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    ***Exception: SegFault         1.09   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         1.20   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      1.15   









::

    opticks-t

    FAILS:  12  / 430   :  Fri Sep 18 22:31:35 2020   
      32 /32  Test #32 : OpticksCoreTest.IntersectSDFTest              ***Exception: SegFault         0.06   

            DONE : prevent this failing for non-existing inputs 

      30 /53  Test #30 : GGeoTest.GPtsTest                             Child aborted***Exception:     0.58   

            Failing on first mm 0  

            2020-09-18 23:51:16.192 INFO  [237539] [Opticks::loadOriginCacheMeta@1853]  gdmlpath 
            2020-09-18 23:51:16.473 INFO  [237539] [main@141]  geolib.nmm 10
            GPtsTest: /home/blyth/opticks/ggeo/tests/GPtsTest.cc:84: void testGPts::init(): Assertion `parts' failed.
            Aborted (core dumped)
           
            #3  0x00007ffff3bf30d2 in __assert_fail () from /lib64/libc.so.6
            #4  0x0000000000405378 in testGPts::init (this=0x7fffffffab00) at /home/blyth/opticks/ggeo/tests/GPtsTest.cc:84
            #5  0x0000000000405307 in testGPts::testGPts (this=0x7fffffffab00, meshlib_=0x636ae0, bndlib_=0xb729d0, mm_=0xcb9930) at /home/blyth/opticks/ggeo/tests/GPtsTest.cc:77
            #6  0x0000000000404032 in main (argc=1, argv=0x7fffffffb1a8) at /home/blyth/opticks/ggeo/tests/GPtsTest.cc:152
            (gdb) 

            GGeoLib::loadConstituents should be loading and associating these     

            GGeoLib=INFO GPtsTest 

            Suspect can no longer do this comparison as the GParts has been dropped ?



      21 /28  Test #21 : OptiXRapTest.interpolationTest                ***Failed                      10.43  

           fails for lack of numpy in the python (juno) picked off PATH
           easy to kludge eg using python3, but what is the definitive solution ?  

           * added SSys::RunPythonScript and SSys:ResolvePython to fix this kind of problem definitively (hopefully)
             by making sensitive to OPTICKS_PYTHON envvar to pick the python

           opticks-c python


      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     9.92   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: SegFault         1.10   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     1.13   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     1.10   
      7  /34  Test #7  : CFG4Test.CG4Test                              ***Exception: SegFault         1.16   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   ***Exception: SegFault         1.13   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    ***Exception: SegFault         1.10   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: SegFault         1.49   



      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      0.34   
    [blyth@localhost opticks]$ date
    Fri Sep 18 22:39:03 CST 2020

