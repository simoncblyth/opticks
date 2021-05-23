G4OpticksRecorder_shakedown
===============================


::

    2021-05-23 15:52:02.790 INFO  [342221] [G4OpticksRecorder::PostUserTrackingAction@103] 
    2021-05-23 15:52:02.790 INFO  [342221] [CManager::PostUserTrackingAction@209] 
    2021-05-23 15:52:02.790 FATAL [342221] [CPhoton::add@100]  _badflag 0
    Assertion failed: (0), function add, file /Users/blyth/opticks/cfg4/CPhoton.cc, line 103.
    Process 33007 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff61275b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff61275b66 <+10>: jae    0x7fff61275b70            ; <+20>
        0x7fff61275b68 <+12>: movq   %rax, %rdi
        0x7fff61275b6b <+15>: jmp    0x7fff6126cae9            ; cerror_nocancel
        0x7fff61275b70 <+20>: retq   
    Target 0: (CerenkovMinimal) stopped.

    Process 33007 launched: '/usr/local/opticks/lib/CerenkovMinimal' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff61275b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff61440080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff611d11ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff611991ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000100259523 libCFG4.dylib`CPhoton::add(this=0x000000010e7db348, flag=0, material=3) at CPhoton.cc:103
        frame #5: 0x000000010025c4b6 libCFG4.dylib`CWriter::writeStepPoint(this=0x000000010e789e40, point=0x000000010ffe44c0, flag=0, material=3, last=false) at CWriter.cc:167
        frame #6: 0x000000010024a592 libCFG4.dylib`CRecorder::WriteStepPoint(this=0x000000010e7db310, point=0x000000010ffe44c0, flag=0, material=3, boundary_status=Undefined, (null)="POST", last=false) at CRecorder.cc:630
        frame #7: 0x00000001002498c6 libCFG4.dylib`CRecorder::postTrackWriteSteps(this=0x000000010e7db310) at CRecorder.cc:550
        frame #8: 0x0000000100248617 libCFG4.dylib`CRecorder::postTrack(this=0x000000010e7db310) at CRecorder.cc:193
        frame #9: 0x000000010028bb9f libCFG4.dylib`CManager::postTrack(this=0x000000011665d0f0) at CManager.cc:236
        frame #10: 0x000000010028bb4d libCFG4.dylib`CManager::PostUserTrackingAction(this=0x000000011665d0f0, track=0x000000010ffe3410) at CManager.cc:218
        frame #11: 0x000000010012ee67 libG4OK.dylib`G4OpticksRecorder::PostUserTrackingAction(this=0x000000010e44c4e0, track=0x000000010ffe3410) at G4OpticksRecorder.cc:104
        frame #12: 0x000000010002ab3f CerenkovMinimal`TrackingAction::PostUserTrackingAction(G4Track const*) + 47
        frame #13: 0x00000001021ba937 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010e4ddf20, apValueG4Track=0x000000010ffe3410) at G4TrackingManager.cc:140
        frame #14: 0x000000010208071a libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010e4dde90, anEvent=0x000000011630dbe0) at G4EventManager.cc:185
        frame #15: 0x0000000102081c2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010e4dde90, anEvent=0x000000011630dbe0) at G4EventManager.cc:338
        frame #16: 0x0000000101f8d9e5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010e6152e0, i_event=0) at G4RunManager.cc:399
        frame #17: 0x0000000101f8d815 libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010e6152e0, n_event=3, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #18: 0x0000000101f8bcd1 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010e6152e0, n_event=3, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #19: 0x000000010003046d CerenkovMinimal`G4::beamOn(int) + 45
        frame #20: 0x0000000100030337 CerenkovMinimal`G4::G4(int) + 1015
        frame #21: 0x000000010003049b CerenkovMinimal`G4::G4(int) + 27
        frame #22: 0x000000010000fa50 CerenkovMinimal`main + 608
        frame #23: 0x00007fff61125015 libdyld.dylib`start + 1
    (lldb) 

