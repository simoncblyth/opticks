OKG4Test_CPhoton_badflag : FIXED : was using the defaul G4OpBoundaryProcess when should use the custom one
============================================================================================================

issue : badflag zeros trips CPhoton assert
--------------------------------------------

::

    2018-08-20 10:10:35.646 FATAL [438239] [CPhoton::add@62]  badflag 0
    Assertion failed: (0), function add, file /Users/blyth/opticks/cfg4/CPhoton.cc, line 65.
    Process 33989 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff60294b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff60294b6e <+10>: jae    0x7fff60294b78            ; <+20>
        0x7fff60294b70 <+12>: movq   %rax, %rdi
        0x7fff60294b73 <+15>: jmp    0x7fff6028bb00            ; cerror_nocancel
        0x7fff60294b78 <+20>: retq   
    Target 0: (OKG4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff60294b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff6045f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff601f01ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff601b81ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106a2c383 libCFG4.dylib`CPhoton::add(this=0x0000000110d0e9b0, flag=0, material=1) at CPhoton.cc:65
        frame #5: 0x0000000106a2f02f libCFG4.dylib`CWriter::writeStepPoint(this=0x0000000110d0eae0, point=0x00000001124d5650, flag=0, material=1) at CWriter.cc:115
        frame #6: 0x0000000106a1d58e libCFG4.dylib`CRecorder::RecordStepPoint(this=0x0000000110d0e970, point=0x00000001124d5650, flag=0, material=1, boundary_status=Undefined, (null)="POST") at CRecorder.cc:467
        frame #7: 0x0000000106a1c8d5 libCFG4.dylib`CRecorder::postTrackWriteSteps(this=0x0000000110d0e970) at CRecorder.cc:403
        frame #8: 0x0000000106a1bcee libCFG4.dylib`CRecorder::postTrack(this=0x0000000110d0e970) at CRecorder.cc:132
        frame #9: 0x0000000106a56551 libCFG4.dylib`CG4::postTrack(this=0x00000001121055c0) at CG4.cc:240
        frame #10: 0x0000000106a50eb7 libCFG4.dylib`CTrackingAction::PostUserTrackingAction(this=0x0000000110d0ec10, track=0x00000001124d4ac0) at CTrackingAction.cc:93
        frame #11: 0x00000001088f2937 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x00000001121fa9a0, apValueG4Track=0x00000001124d4ac0) at G4TrackingManager.cc:140
        frame #12: 0x00000001087b971a libG4event.dylib`G4EventManager::DoProcessing(this=0x00000001121fa910, anEvent=0x00000001124d2250) at G4EventManager.cc:185
        frame #13: 0x00000001087bac2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x00000001121fa910, anEvent=0x00000001124d2250) at G4EventManager.cc:338
        frame #14: 0x00000001086c69f5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000112105770, i_event=0) at G4RunManager.cc:399
        frame #15: 0x00000001086c6825 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000112105770, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #16: 0x00000001086c4ce1 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000112105770, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #17: 0x0000000106a571d9 libCFG4.dylib`CG4::propagate(this=0x00000001121055c0) at CG4.cc:315
        frame #18: 0x00000001000df2b6 libOKG4.dylib`OKG4Mgr::propagate_(this=0x00007ffeefbfe918) at OKG4Mgr.cc:141
        frame #19: 0x00000001000deec6 libOKG4.dylib`OKG4Mgr::propagate(this=0x00007ffeefbfe918) at OKG4Mgr.cc:84
        frame #20: 0x00000001000148b9 OKG4Test`main(argc=5, argv=0x00007ffeefbfea00) at OKG4Test.cc:9
        frame #21: 0x00007fff60144015 libdyld.dylib`start + 1
    (lldb) 


Dumping reveals are always getting boundary status Undefined, which explains all the bad flags::

    2018-08-20 13:35:24.152 INFO  [72450] [CG4Ctx::setStepOptical@291]  _prior_boundary_status                           Undefined _boundary_status                           Undefined
    2018-08-20 13:35:24.152 INFO  [72450] [CG4Ctx::setStepOptical@291]  _prior_boundary_status                           Undefined _boundary_status                           Undefined

::

    (lldb) b CBoundaryProcess::GetOpBoundaryProcessStatus
    Breakpoint 1: where = libCFG4.dylib`CBoundaryProcess::GetOpBoundaryProcessStatus() + 4 at CBoundaryProcess.cc:7, address = 0x000000000011b594
    (lldb) 




Huh no boundary process in procvec ?::

    (lldb) c
    Process 6609 resuming
    Process 6609 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 4.1
        frame #0: 0x0000000106a50628 libCFG4.dylib`CBoundaryProcess::GetOpBoundaryProcessStatus() at CBoundaryProcess.cc:15
       12  	        DsG4OpBoundaryProcess* opProc = NULL ;  
       13  	        G4int npmax = mgr->GetPostStepProcessVector()->entries();
       14  	        G4ProcessVector* pv = mgr->GetPostStepProcessVector(typeDoIt);
    -> 15  	        for (G4int i=0; i<npmax; i++) 
       16  	        {
       17  	            G4VProcess* proc = (*pv)[i];
       18  	            opProc = dynamic_cast<DsG4OpBoundaryProcess*>(proc);
    Target 0: (OKG4Test) stopped.
    (lldb) p pv
    (G4ProcessVector *) $2 = 0x0000000110d12c70
    (lldb) p *pv
    (G4ProcessVector) $3 = {
      pProcVector = 0x0000000110d12c80 size=1
    }
    (lldb) b 18
    Breakpoint 5: where = libCFG4.dylib`CBoundaryProcess::GetOpBoundaryProcessStatus() + 190 at CBoundaryProcess.cc:18, address = 0x0000000106a5064e
    (lldb) c
    Process 6609 resuming
    Process 6609 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 5.1
        frame #0: 0x0000000106a5064e libCFG4.dylib`CBoundaryProcess::GetOpBoundaryProcessStatus() at CBoundaryProcess.cc:18
       15  	        for (G4int i=0; i<npmax; i++) 
       16  	        {
       17  	            G4VProcess* proc = (*pv)[i];
    -> 18  	            opProc = dynamic_cast<DsG4OpBoundaryProcess*>(proc);
       19  	            if (opProc) 
       20  	            { 
       21  	                status = opProc->GetStatus(); 
    Target 0: (OKG4Test) stopped.
    (lldb) p proc
    (G4Transportation *) $4 = 0x0000000110d154a0
    (lldb) 





