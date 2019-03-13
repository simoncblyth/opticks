ckm_revival : improve CerenkovMinimal handling of missing RNG seq 
===================================================================

Objectives
----------

1. get back into context, by reviving CerenkovMinimal 
2. review completeness of G4OK interface


Mar 13, 2019 : ckm-run : aborts for lack of RNG
--------------------------------------------------

::

    epsilon:~ blyth$ t ckm-run
    ckm-run is a function
    ckm-run () 
    { 
        g4-;
        g4-export;
        CerenkovMinimal
    }


::

    2019-03-13 13:42:06.654 FATAL [1344679] [G4Opticks::setGeometry@136] ]]]
    2019-03-13 13:42:06.655 INFO  [1344679] [SensitiveDetector::Initialize@103]  HCE 0x7f8514466dc0 HCE.Capacity 2 SensitiveDetectorName SD0 collectionName[0] OpHitCollectionA collectionName[1] OpHitCollectionB
    2019-03-13 13:42:06.658 FATAL [1344679] [Ctx::setTrack@67]  _track_particle_name e+ _track_id 0 _step_id -1 num_gs 0 max_gs 1 kill 0
    2019-03-13 13:42:06.660 VERB  [1344679] [*L4Cerenkov::PostStepDoIt@303] 1.9152e-05
    Assertion failed: (seq_index < m_seq_ni), function setSequenceIndex, file /Users/blyth/opticks/cfg4/CAlignEngine.cc, line 113.
    Abort trap: 6


Improve the error handling of missing RNG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::
   cfg4--   ## rebuild just CFG4 lib

   epsilon:cfg4 blyth$ lldb CerenkovMinimal


    2019-03-13 13:54:08.479 FATAL [1353533] [G4Opticks::setGeometry@136] ]]]
    2019-03-13 13:54:08.479 INFO  [1353533] [SensitiveDetector::Initialize@103]  HCE 0x110bc1a60 HCE.Capacity 2 SensitiveDetectorName SD0 collectionName[0] OpHitCollectionA collectionName[1] OpHitCollectionB
    2019-03-13 13:54:08.480 FATAL [1353533] [Ctx::setTrack@67]  _track_particle_name e+ _track_id 0 _step_id -1 num_gs 0 max_gs 1 kill 0
    2019-03-13 13:54:08.480 VERB  [1353533] [*L4Cerenkov::PostStepDoIt@303] 1.9152e-05
    2019-03-13 13:54:08.480 FATAL [1353533] [CAlignEngine::setSequenceIndex@114] OUT OF RANGE : CAlignEngine seq_index -1 seq - seq_ni 0 seq_nv 0 cur 0 seq_path $TMP/TRngBufTest.npy simstream logpath /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/CAlignEngine.log recycle_idx 0
    Assertion failed: (have_seq), function setSequenceIndex, file /Users/blyth/opticks/cfg4/CAlignEngine.cc, line 115.
    Process 15299 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff65123b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff65123b66 <+10>: jae    0x7fff65123b70            ; <+20>
        0x7fff65123b68 <+12>: movq   %rax, %rdi
        0x7fff65123b6b <+15>: jmp    0x7fff6511aae9            ; cerror_nocancel
        0x7fff65123b70 <+20>: retq   
    Target 0: (CerenkovMinimal) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff65123b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff652ee080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff6507f1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff650471ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010026fb0c libCFG4.dylib`CAlignEngine::setSequenceIndex(this=0x0000000110b2de00, seq_index=0) at CAlignEngine.cc:115
        frame #5: 0x000000010026f94a libCFG4.dylib`CAlignEngine::SetSequenceIndex(seq_index=0) at CAlignEngine.cc:33
        frame #6: 0x0000000100119057 libG4OK.dylib`G4Opticks::setAlignIndex(this=0x0000000110e4f010, align_idx=0) const at G4Opticks.cc:204
        frame #7: 0x0000000100020a88 CerenkovMinimal`L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&) + 2840
        frame #8: 0x00000001024007db libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x0000000110d69820, np=5) at G4SteppingManager2.cc:538
        frame #9: 0x000000010240064d libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x0000000110d69820) at G4SteppingManager2.cc:510
        frame #10: 0x00000001023fbdaa libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000110d69820) at G4SteppingManager.cc:209
        frame #11: 0x000000010241286f libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000110d697e0, apValueG4Track=0x0000000110bc28c0) at G4TrackingManager.cc:126
        frame #12: 0x00000001022d971a libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000110d69750, anEvent=0x0000000110bc0e50) at G4EventManager.cc:185
        frame #13: 0x00000001022dac2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x0000000110d69750, anEvent=0x0000000110bc0e50) at G4EventManager.cc:338
        frame #14: 0x00000001021e69f5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000110c5dd00, i_event=0) at G4RunManager.cc:399
        frame #15: 0x00000001021e6825 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000110c5dd00, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #16: 0x00000001021e4ce1 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000110c5dd00, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #17: 0x0000000100032cbd CerenkovMinimal`G4::beamOn(int) + 45
        frame #18: 0x0000000100032b67 CerenkovMinimal`G4::G4(int) + 1015
        frame #19: 0x0000000100032ceb CerenkovMinimal`G4::G4(int) + 27
        frame #20: 0x0000000100010ff6 CerenkovMinimal`main + 550
        frame #21: 0x00007fff64fd3015 libdyld.dylib`start + 1
        frame #22: 0x00007fff64fd3015 libdyld.dylib`start + 1
    (lldb) 


examples/Geant4/CerenkovMinimal/L4Cerenkov.cc setAlignIndex is teeing up the random stream in the CPU photon generation loop::

    343     for (G4int i = 0; i < NumPhotons; i++) {
    344 
    345         // Determine photon energy
    346 #ifdef WITH_OPTICKS
    347         unsigned record_id = opticks_photon_offset+i ;
    348         G4Opticks::GetOpticks()->setAlignIndex(record_id);
    349 #endif
    350 
    351         G4double rand;
    352         G4double sampledEnergy, sampledRI;
    353         G4double cosTheta, sin2Theta;
    354 
    355         // sample an energy
    356 
    357         do {
    358             rand = G4UniformRand();
    359             sampledEnergy = Pmin + rand * dp;
    ...
 


Cause of this issue is that *$TMP/TRngBufTest.npy* lives in temporary dirs.::

    epsilon:CerenkovMinimal blyth$ opticks-find TRngBufTest.npy
    ./thrustrap/tests/TRngTest.cu:    const char* path = "$TMP/TRngBufTest.npy" ; 
    ./thrustrap/tests/TRngTest.cu:    //  import os, numpy as np ; a = np.load(os.path.expandvars("$TMP/TRngBufTest.npy"))
    ./thrustrap/tests/TRngBufTest.cu:    const char* path = "$TMP/TRngBufTest.npy" ; 
    ./thrustrap/tests/TRngBufTest.cu:    //  import os, numpy as np ; a = np.load(os.path.expandvars("$TMP/TRngBufTest.npy"))
    ./examples/UseThrustRap/UseThrustRap.cu:    const char* path = "$TMP/TRngBufTest.npy" ; 
    ./examples/UseThrustRap/UseThrustRap.cu:    //  import os, numpy as np ; a = np.load(os.path.expandvars("$TMP/TRngBufTest.npy"))
    ./cfg4/CAlignEngine.cc:    m_seq_path("$TMP/TRngBufTest.npy"),
    ./cfg4/CRandomEngine.cc:    m_path("$TMP/TRngBufTest.npy"),
    ./ana/ucf.py:        return os.path.expandvars("$TMP/TRngBufTest.npy" )
    ./thrustrap/tests/TRngBufTest.py:    a = np.load(os.path.expandvars("$TMP/TRngBufTest.npy"))
    epsilon:opticks blyth$ 



