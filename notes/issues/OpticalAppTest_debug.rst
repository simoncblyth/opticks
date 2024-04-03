OpticalAppTest_debug
======================


Avoid "the Momentum Change is not unit vector !!  Difference:  5.45723e-08"  precision warning with unit::

    233     double phi = 2.*M_PI*frac ;
    234     double sinPhi = sinf(phi);
    235     double cosPhi = cosf(phi);
    236 
    237     G4ThreeVector direction(-cosPhi,0.,-sinPhi) ;
    238     direction = direction.unit();
    239 


::

    OpticalRecorder::PreUserTrackingAction trk_idx :99 point_idx:0
      G4ParticleChange::CheckIt  : the Momentum Change is not unit vector !!  Difference:  5.45723e-08
    opticalphoton E=2.952e-06 pos=0.0499753, 0, -0.00157053
    Process 64991 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x0000000104df5259 libG4track.dylib`G4ParticleChange::DumpInfo(this=0x0000000106706a40) const at G4ParticleChange.cc:448
       445 	void G4ParticleChange::DumpInfo() const
       446 	{
       447 	// use base-class DumpInfo
    -> 448 	  G4VParticleChange::DumpInfo();
       449 	
       450 	  G4int oldprc = G4cout.precision(3);
       451 	
    Target 0: (OpticalAppTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000104df5259 libG4track.dylib`G4ParticleChange::DumpInfo(this=0x0000000106706a40) const at G4ParticleChange.cc:448
        frame #1: 0x0000000104dfee3b libG4track.dylib`G4ParticleChangeForTransport::DumpInfo(this=0x0000000106706a40) const at G4ParticleChangeForTransport.cc:258
        frame #2: 0x0000000104df76cc libG4track.dylib`G4ParticleChange::CheckIt(this=0x0000000106706a40, aTrack=0x000000010671b280) at G4ParticleChange.cc:658
        frame #3: 0x0000000104dfead4 libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(this=0x0000000106706a40, pStep=0x0000000106a2dc70) at G4ParticleChangeForTransport.cc:202
        frame #4: 0x0000000101d98ed8 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x0000000106a2dae0) at G4SteppingManager2.cc:424
        frame #5: 0x0000000101d94c91 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000106a2dae0) at G4SteppingManager.cc:191
        frame #6: 0x0000000101dab86f libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000106a2daa0, apValueG4Track=0x000000010671b280) at G4TrackingManager.cc:126
        frame #7: 0x0000000101c7171a libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000106a2da10, anEvent=0x00000001067079e0) at G4EventManager.cc:185
        frame #8: 0x0000000101c72c2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x0000000106a2da10, anEvent=0x00000001067079e0) at G4EventManager.cc:338
        frame #9: 0x0000000101b7e9e5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000106a2d830, i_event=0) at G4RunManager.cc:399
        frame #10: 0x0000000101b7e815 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000106a2d830, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #11: 0x0000000101b7ccd1 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000106a2d830, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #12: 0x00000001000047b3 OpticalAppTest`OpticalApp::Main() at OpticalApp.h:88
        frame #13: 0x0000000100006924 OpticalAppTest`main at OpticalAppTest.cc:4
        frame #14: 0x00007fff50548015 libdyld.dylib`start + 1
        frame #15: 0x00007fff50548015 libdyld.dylib`start + 1
    (lldb) 


