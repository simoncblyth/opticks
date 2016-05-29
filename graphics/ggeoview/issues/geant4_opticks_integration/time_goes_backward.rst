Time Goes Backward
===================

[FIXED] Issue : with GDML geometry opticalphoton time going backwards
----------------------------------------------------------------------

Problem was the use of spline interpolation with non-smooth enough ABSLENGTH
resulting in negative absorption lengths. 

For debug session details see issues/optical_local_time_goes_backward.rst


::
  
    ggv-;ggv-g4gun

* maybe material props (eg refractive index) are messed up 


::

    *** G4Exception : TRACK003
          issued by : G4ParticleChange::CheckIt
    momentum, energy, and/or time was illegal
    *** Event Must Be Aborted ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------

      G4VParticleChange::CheckIt    : the true step length is negative  !!  Difference:  0.0577914[MeV] 
    opticalphoton E=6.60373e-06 pos=-18.0795, -799.699, -6.60502


    (lldb) b "G4ParticleChange::CheckIt(G4Track const&)" 
           b 546  # once inside   

    (lldb) bt
    * thread #1: tid = 0x61a7be, 0x0000000105e901cd libG4track.dylib`G4ParticleChange::CheckIt(this=0x00000001091e8cc0, aTrack=0x000000010ec024b0) + 909 at G4ParticleChange.cc:546, queue = 'com.apple.main-thread', stop reason = breakpoint 4.1
      * frame #0: 0x0000000105e901cd libG4track.dylib`G4ParticleChange::CheckIt(this=0x00000001091e8cc0, aTrack=0x000000010ec024b0) + 909 at G4ParticleChange.cc:546
        frame #1: 0x0000000105e9963f libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(this=0x00000001091e8cc0, pStep=0x00000001091270d0) + 1519 at G4ParticleChangeForTransport.cc:202
        frame #2: 0x0000000102e8896e libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x0000000109126f40) + 254 at G4SteppingManager2.cc:420
        frame #3: 0x0000000102e84168 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000109126f40) + 504 at G4SteppingManager.cc:191
        frame #4: 0x0000000102e9b92d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000109126f00, apValueG4Track=0x000000010ec024b0) + 1357 at G4TrackingManager.cc:126
        frame #5: 0x0000000102d78e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000109126e70, anEvent=0x000000010eb808d0) + 3188 at G4EventManager.cc:185
        frame #6: 0x0000000102d79b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x0000000109126e70, anEvent=0x000000010eb808d0) + 47 at G4EventManager.cc:336
        frame #7: 0x0000000102ca6c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x00000001087234e0, i_event=0) + 69 at G4RunManager.cc:399
        frame #8: 0x0000000102ca6ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x00000001087234e0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #9: 0x0000000102ca58e4 libG4run.dylib`G4RunManager::BeamOn(this=0x00000001087234e0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #10: 0x000000010153edb0 libcfg4.dylib`CG4::propagate(this=0x0000000108721330) + 752 at CG4.cc:137
        frame #11: 0x000000010000d5a2 CG4Test`main(argc=11, argv=0x00007fff5fbfdd50) + 210 at CG4Test.cc:18
        frame #12: 0x00007fff89e755fd libdyld.dylib`start + 1



