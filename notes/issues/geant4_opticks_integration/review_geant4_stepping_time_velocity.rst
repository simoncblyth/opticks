Review Geant4 Stepping Time Velocity 
======================================

where does G4 set times anyhow ?
----------------------------------

Recorded time comes from::

     820 void CRecorder::RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* /*label*/ )
     821 {
     822     const G4ThreeVector& pos = point->GetPosition();
     823     const G4ThreeVector& pol = point->GetPolarization();
     824 
     825     G4double time = point->GetGlobalTime();

::

    delta:cfg4 blyth$ g4-cc SetGlobalTime
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/models/fission/src/G4FissionLibrary.cc://    it->SetGlobalTime(fe->getNeutronAge(i)*second);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/models/fission/src/G4FissionLibrary.cc://    it->SetGlobalTime(fe->getPhotonAge(i)*second);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/stopping/src/G4HadronStoppingProcess.cc:  thePro.SetGlobalTime(0.0);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/stopping/src/G4HadronStoppingProcess.cc:    thePro.SetGlobalTime(0.0);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/stopping/src/G4MuonMinusBoundDecay.cc:  p->SetGlobalTime(time);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/parameterisation/src/G4FastStep.cc:  pPostStepPoint->SetGlobalTime( theTimeChange  );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/parameterisation/src/G4FastStep.cc:  pPostStepPoint->SetGlobalTime( theTimeChange  );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForDecay.cc:  pPostStepPoint->SetGlobalTime( GetGlobalTime() );
    delta:cfg4 blyth$ 
    delta:cfg4 blyth$ g4-hh SetGlobalTime
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/util/include/G4HadProjectile.hh:  inline void SetGlobalTime(G4double t);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/hadronic/util/include/G4HadProjectile.hh:inline void G4HadProjectile::SetGlobalTime(G4double t) 
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4StepPoint.hh:   void SetGlobalTime(const G4double aValue);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4Track.hh:   void SetGlobalTime(const G4double aValue);
    delta:cfg4 blyth$ 
    delta:cfg4 blyth$ g4-icc SetGlobalTime
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4Step.icc:   fpPreStepPoint->SetGlobalTime(fpTrack->GetGlobalTime());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4Step.icc:   fpTrack->SetGlobalTime(fpPostStepPoint->GetGlobalTime());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4StepPoint.icc: void G4StepPoint::SetGlobalTime(const G4double aValue)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/include/G4Track.icc:   inline void G4Track::SetGlobalTime(const G4double aValue)
    delta:cfg4 blyth$ 



G4Step 
* postStep -> preStep  ?? at each step

* initialize preStepPoint from the track
* updateTrack from the postStepPoint 

::


      track  --->  preStep
           ^
            \
             \___  postStep




G4TrackingManager::ProcessOneTrack
-------------------------------------

::

    067 void G4TrackingManager::ProcessOneTrack(G4Track* apValueG4Track)
     68 ////////////////////////////////////////////////////////////////
     69 {
     70 
     71   // Receiving a G4Track from the EventManager, this funciton has the
     72   // responsibility to trace the track till it stops.
     73   fpTrack = apValueG4Track;
    ...
    ...     clear secondaries
    ...
     86   if(verboseLevel>0 && (G4VSteppingVerbose::GetSilent()!=1) ) TrackBanner();
     87  
     88   // Give SteppingManger the pointer to the track which will be tracked 
     89   fpSteppingManager->SetInitialStep(fpTrack);
    ///
    ///       Only call to G4StepPoint::SetGlobalTime happens in here
    ///
     90 
     91   // Pre tracking user intervention process.
     92   fpTrajectory = 0;
     93   if( fpUserTrackingAction != 0 ) {
     94      fpUserTrackingAction->PreUserTrackingAction(fpTrack);
     95   }
    ...  
    ...    trajectory setup
    ...
    110   // Give SteppingManger the maxmimum number of processes 
    111   fpSteppingManager->GetProcessNumber();
    112 
    113   // Give track the pointer to the Step
    114   fpTrack->SetStep(fpSteppingManager->GetStep());
    115 
    116   // Inform beginning of tracking to physics processes 
    117   fpTrack->GetDefinition()->GetProcessManager()->StartTracking(fpTrack);
    118 
    119   // Track the particle Step-by-Step while it is alive
    120   //  G4StepStatus stepStatus;
    121 
    122   while( (fpTrack->GetTrackStatus() == fAlive) ||
    123          (fpTrack->GetTrackStatus() == fStopButAlive) ){
    124 
    125     fpTrack->IncrementCurrentStepNumber();
    126     fpSteppingManager->Stepping();
    127 #ifdef G4_STORE_TRAJECTORY
    128     if(StoreTrajectory) fpTrajectory->
    129                         AppendStep(fpSteppingManager->GetStep());
    130 #endif
    131     if(EventIsAborted) {
    132       fpTrack->SetTrackStatus( fKillTrackAndSecondaries );
    133     }
    134   }
    135   // Inform end of tracking to physics processes 
    136   fpTrack->GetDefinition()->GetProcessManager()->EndTracking();
    137 
    138   // Post tracking user intervention process.
    139   if( fpUserTrackingAction != 0 ) {
    140      fpUserTrackingAction->PostUserTrackingAction(fpTrack);
    141   }
    ...       trajectory cleanup
    150   }
    151 }



G4Step::InitializeStep Track->PreStepPoint
-----------------------------------------------

::

   (lldb) b G4StepPoint::SetGlobalTime

Step point time only ever set at initialization, from the track::

    (lldb) bt
    * thread #1: tid = 0x1059dc, 0x0000000104c753a9 libG4tracking.dylib`G4Step::InitializeStep(G4Track*) [inlined] G4StepPoint::SetGlobalTime(this=0x000000011127a650, aValue=<unavailable>) at G4StepPoint.icc:60, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000104c753a9 libG4tracking.dylib`G4Step::InitializeStep(G4Track*) [inlined] G4StepPoint::SetGlobalTime(this=0x000000011127a650, aValue=<unavailable>) at G4StepPoint.icc:60
        frame #1: 0x0000000104c753a9 libG4tracking.dylib`G4Step::InitializeStep(this=0x000000011127a5f0, aValue=0x000000012818a7b0) + 89 at G4Step.icc:200
        frame #2: 0x0000000104c7502c libG4tracking.dylib`G4SteppingManager::SetInitialStep(this=0x000000011127a460, valueTrack=<unavailable>) + 1468 at G4SteppingManager.cc:356
        frame #3: 0x0000000104c7e4a7 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000011127a420, apValueG4Track=<unavailable>) + 199 at G4TrackingManager.cc:89
        frame #4: 0x0000000104bd6727 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000011127a390, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #5: 0x0000000104b58611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010f66ef00, i_event=0) + 49 at G4RunManager.cc:399
        frame #6: 0x0000000104b584db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010f66ef00, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #7: 0x0000000104b57913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010f66ef00, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #8: 0x0000000103ee4882 libcfg4.dylib`CG4::propagate(this=0x000000010f66ee50) + 1458 at CG4.cc:270
        frame #9: 0x0000000103fcd52a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfe3f0) + 538 at OKG4Mgr.cc:82
        frame #10: 0x00000001000139ca OKG4Test`main(argc=29, argv=0x00007fff5fbfe4d0) + 1498 at OKG4Test.cc:57
        frame #11: 0x00007fff915315fd libdyld.dylib`start + 1
    (lldb) f 2
    frame #2: 0x0000000104c7502c libG4tracking.dylib`G4SteppingManager::SetInitialStep(this=0x000000011127a460, valueTrack=<unavailable>) + 1468 at G4SteppingManager.cc:356
       353     }
       354     else {
       355  // Initial set up for attribues of 'Step'
    -> 356         fStep->InitializeStep( fTrack );
       357     }
       358  #ifdef G4VERBOSE
       359                           // !!!!! Verbose
    (lldb) f 1
    frame #1: 0x0000000104c753a9 libG4tracking.dylib`G4Step::InitializeStep(this=0x000000011127a5f0, aValue=0x000000012818a7b0) + 89 at G4Step.icc:200
       197     // To avoid the circular dependency between G4Track, G4Step
       198     // and G4StepPoint, G4Step has to manage the copy actions.
       199     fpPreStepPoint->SetPosition(fpTrack->GetPosition());
    -> 200     fpPreStepPoint->SetGlobalTime(fpTrack->GetGlobalTime());
       201     fpPreStepPoint->SetLocalTime(fpTrack->GetLocalTime());
       202     fpPreStepPoint->SetProperTime(fpTrack->GetProperTime());
       203     fpPreStepPoint->SetMomentumDirection(fpTrack->GetMomentumDirection());
    (lldb) 


::

    184 inline
    185  void G4Step::InitializeStep( G4Track* aValue )
    186  {
    187    // Initialize G4Step attributes
    188    fStepLength = 0.;
    189    fTotalEnergyDeposit = 0.;
    190    fNonIonizingEnergyDeposit = 0.;
    191    fpTrack = aValue;
    192    fpTrack->SetStepLength(0.);
    193 
    194    nSecondaryByLastStep = 0;
    195 
    196    // Initialize G4StepPoint attributes.
    197    // To avoid the circular dependency between G4Track, G4Step
    198    // and G4StepPoint, G4Step has to manage the copy actions.
    199    fpPreStepPoint->SetPosition(fpTrack->GetPosition());
    200    fpPreStepPoint->SetGlobalTime(fpTrack->GetGlobalTime());
    201    fpPreStepPoint->SetLocalTime(fpTrack->GetLocalTime());
    202    fpPreStepPoint->SetProperTime(fpTrack->GetProperTime());
    203    fpPreStepPoint->SetMomentumDirection(fpTrack->GetMomentumDirection());
    204    fpPreStepPoint->SetKineticEnergy(fpTrack->GetKineticEnergy());
    205    fpPreStepPoint->SetTouchableHandle(fpTrack->GetTouchableHandle());
    206    fpPreStepPoint->SetMaterial( fpTrack->GetTouchable()->GetVolume()->GetLogicalVolume()->GetMaterial());
    207    fpPreStepPoint->SetMaterialCutsCouple( fpTrack->GetTouchable()->GetVolume()->GetLogicalVolume()->GetMaterialCutsCouple());
    208    fpPreStepPoint->SetSensitiveDetector( fpTrack->GetTouchable()->GetVolume()->GetLogicalVolume()->GetSensitiveDetector());
    209    fpPreStepPoint->SetPolarization(fpTrack->GetPolarization());
    210    fpPreStepPoint->SetSafety(0.);
    211    fpPreStepPoint->SetStepStatus(fUndefined);
    212    fpPreStepPoint->SetProcessDefinedStep(0);
    213    fpPreStepPoint->SetMass(fpTrack->GetDynamicParticle()->GetMass());
    214    fpPreStepPoint->SetCharge(fpTrack->GetDynamicParticle()->GetCharge());
    215    fpPreStepPoint->SetWeight(fpTrack->GetWeight());
    216 
    217    // Set Velocity
    218    //  should be placed after SetMaterial for preStep point
    219     fpPreStepPoint->SetVelocity(fpTrack->CalculateVelocity());
    220 
    221    (*fpPostStepPoint) = (*fpPreStepPoint);
    222  }




G4StepPoint::AddGlobalTime
-----------------------------

::

    (lldb) bt
    * thread #1: tid = 0x121f8c, 0x0000000106ae4d38 libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(G4Step*) [inlined] G4StepPoint::AddGlobalTime(this=<unavailable>, aValue=<unavailable>) at G4StepPoint.icc:64, queue = 'com.apple.main-thread', stop reason = breakpoint 2.4
      * frame #0: 0x0000000106ae4d38 libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(G4Step*) [inlined] G4StepPoint::AddGlobalTime(this=<unavailable>, aValue=<unavailable>) at G4StepPoint.icc:64
        frame #1: 0x0000000106ae4d38 libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(this=0x0000000110a79180, pStep=0x0000000110a28d90) + 488 at G4ParticleChangeForTransport.cc:194
        frame #2: 0x0000000104c769b4 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x0000000110a28c00) + 116 at G4SteppingManager2.cc:420
        frame #3: 0x0000000104c74771 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000110a28c00) + 417 at G4SteppingManager.cc:191
        frame #4: 0x0000000104c7e771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000110a28bc0, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #5: 0x0000000104bd6727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000110a28b30, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #6: 0x0000000104b58611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010e61d630, i_event=0) + 49 at G4RunManager.cc:399
        frame #7: 0x0000000104b584db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010e61d630, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #8: 0x0000000104b57913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010e61d630, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #9: 0x0000000103ee4842 libcfg4.dylib`CG4::propagate(this=0x000000010e61d580) + 1458 at CG4.cc:270
        frame #10: 0x0000000103fcd52a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfe3e0) + 538 at OKG4Mgr.cc:82
        frame #11: 0x00000001000139ca OKG4Test`main(argc=30, argv=0x00007fff5fbfe4c0) + 1498 at OKG4Test.cc:57
        frame #12: 0x00007fff915315fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 1
    frame #1: 0x0000000106ade947 libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(this=0x0000000110aa0d88, pStep=0x0000000110a28d90) + 263 at G4ParticleChange.cc:384
       381        
       382    // update position and time
       383    pPostStepPoint->SetPosition( thePositionChange  );
    -> 384    pPostStepPoint->AddGlobalTime(theTimeChange - theLocalTime0);
       385    pPostStepPoint->SetLocalTime( theTimeChange );           
       386    pPostStepPoint->SetProperTime( theProperTimeChange  );
       387  
    (lldb) 

    (lldb) p theTimeChange
    (G4double) $0 = 12.281634323921269
    (lldb) p theLocalTime0
    (G4double) $1 = 12.281634323921269
    (lldb) p thePositionChange
    (G4ThreeVector) $2 = (dx = 2389.0136288099911, dy = 0, dz = 0)
    (lldb) 


    (lldb) bt
    * thread #1: tid = 0x121f8c, 0x0000000106ade947 libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(G4Step*) [inlined] G4StepPoint::AddGlobalTime(this=0x0000000110a28ec0, aValue=<unavailable>) at G4StepPoint.icc:64, queue = 'com.apple.main-thread', stop reason = breakpoint 2.2
      * frame #0: 0x0000000106ade947 libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(G4Step*) [inlined] G4StepPoint::AddGlobalTime(this=0x0000000110a28ec0, aValue=<unavailable>) at G4StepPoint.icc:64
        frame #1: 0x0000000106ade947 libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(this=0x0000000110aa22e8, pStep=0x0000000110a28d90) + 263 at G4ParticleChange.cc:384
        frame #2: 0x0000000104c76e3c libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x0000000110a28c00, np=<unavailable>) + 76 at G4SteppingManager2.cc:533
        frame #3: 0x0000000104c76d2b libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x0000000110a28c00) + 139 at G4SteppingManager2.cc:502
        frame #4: 0x0000000104c74909 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000110a28c00) + 825 at G4SteppingManager.cc:209
        frame #5: 0x0000000104c7e771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000110a28bc0, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #6: 0x0000000104bd6727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000110a28b30, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #7: 0x0000000104b58611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010e61d630, i_event=0) + 49 at G4RunManager.cc:399
        frame #8: 0x0000000104b584db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010e61d630, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #9: 0x0000000104b57913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010e61d630, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #10: 0x0000000103ee4842 libcfg4.dylib`CG4::propagate(this=0x000000010e61d580) + 1458 at CG4.cc:270
        frame #11: 0x0000000103fcd52a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfe3e0) + 538 at OKG4Mgr.cc:82
        frame #12: 0x00000001000139ca OKG4Test`main(argc=30, argv=0x00007fff5fbfe4c0) + 1498 at OKG4Test.cc:57
        frame #13: 0x00007fff915315fd libdyld.dylib`start + 1
    (lldb) f 1
    frame #1: 0x0000000106ade947 libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(this=0x0000000110aa22e8, pStep=0x0000000110a28d90) + 263 at G4ParticleChange.cc:384
       381        
       382    // update position and time
       383    pPostStepPoint->SetPosition( thePositionChange  );
    -> 384    pPostStepPoint->AddGlobalTime(theTimeChange - theLocalTime0);
       385    pPostStepPoint->SetLocalTime( theTimeChange );           
       386    pPostStepPoint->SetProperTime( theProperTimeChange  );
       387  
    (lldb) p theTimeChange
    (G4double) $3 = 12.281634323921269
    (lldb) p theLocalTime0
    (G4double) $4 = 12.281634323921269
    (lldb) p thePositionChange
    (G4ThreeVector) $5 = (dx = 2389.0136288099911, dy = 0, dz = 0)
    (lldb) 





G4Step::UpdateTrack PostStepPoint->Track
---------------------------------------------


::

    224 inline
    225  void G4Step::UpdateTrack( )
    226  {
    227    // To avoid the circular dependency between G4Track, G4Step
    228    // and G4StepPoint, G4Step has to manage the update actions.
    229    //  position, time
    230    fpTrack->SetPosition(fpPostStepPoint->GetPosition());
    231    fpTrack->SetGlobalTime(fpPostStepPoint->GetGlobalTime());
    232    fpTrack->SetLocalTime(fpPostStepPoint->GetLocalTime());
    233    fpTrack->SetProperTime(fpPostStepPoint->GetProperTime());
    234    //  energy, momentum, polarization
    235    fpTrack->SetMomentumDirection(fpPostStepPoint->GetMomentumDirection());
    236    fpTrack->SetKineticEnergy(fpPostStepPoint->GetKineticEnergy());
    237    fpTrack->SetPolarization(fpPostStepPoint->GetPolarization());
    238    //  mass charge
    239    G4DynamicParticle* pParticle = (G4DynamicParticle*)(fpTrack->GetDynamicParticle());
    240    pParticle->SetMass(fpPostStepPoint->GetMass());
    241    pParticle->SetCharge(fpPostStepPoint->GetCharge());
    242    //  step length
    243    fpTrack->SetStepLength(fStepLength);
    244    // NextTouchable is updated
    245    // (G4Track::Touchable points touchable of Pre-StepPoint)
    246    fpTrack->SetNextTouchableHandle(fpPostStepPoint->GetTouchableHandle());
    247    fpTrack->SetWeight(fpPostStepPoint->GetWeight());
    248 
    249 
    250    // set velocity
    251    fpTrack->SetVelocity(fpPostStepPoint->GetVelocity());
    252 }

Transportation time setting based on velocity and step length

Breakpoint here is good for seeing track step by step::

    b G4Transportation::AlongStepDoIt

::

    525 G4VParticleChange* G4Transportation::AlongStepDoIt( const G4Track& track,
    526                                                     const G4Step&  stepData )
    527 {
    528   static G4ThreadLocal G4int noCalls=0;
    529   noCalls++;
    530 
    531   fParticleChange.Initialize(track) ;
    532 
    533   //  Code for specific process 
    534   //
    535   fParticleChange.ProposePosition(fTransportEndPosition) ;
    536   fParticleChange.ProposeMomentumDirection(fTransportEndMomentumDir) ;
    537   fParticleChange.ProposeEnergy(fTransportEndKineticEnergy) ;
    538   fParticleChange.SetMomentumChanged(fMomentumChanged) ;
    539 
    540   fParticleChange.ProposePolarization(fTransportEndSpin);
    541  
    542   G4double deltaTime = 0.0 ;
    543 
    544   // Calculate  Lab Time of Flight (ONLY if field Equations used it!)
    545   // G4double endTime   = fCandidateEndGlobalTime;
    546   // G4double delta_time = endTime - startTime;
    547 
    548   G4double startTime = track.GetGlobalTime() ;
    549  
    550   if (!fEndGlobalTimeComputed)
    ////
    ////    fEndGlobalTimeComputed always false without magnetic field ???
    ////    THIS LOOKS TO BE WHERE THE TIMES ARE COMING FROM
    ////    USING prestep point velocity and steplength
    ////
    551   {
    552      // The time was not integrated .. make the best estimate possible
    553      //
    554      G4double initialVelocity = stepData.GetPreStepPoint()->GetVelocity();
    555      G4double stepLength      = track.GetStepLength();
    556 
    557      deltaTime= 0.0;  // in case initialVelocity = 0 
    558      if ( initialVelocity > 0.0 )  { deltaTime = stepLength/initialVelocity; }
    559 
    560      fCandidateEndGlobalTime   = startTime + deltaTime ;
    561      fParticleChange.ProposeLocalTime(  track.GetLocalTime() + deltaTime) ;
    562   }
    563   else
    564   {
    565      deltaTime = fCandidateEndGlobalTime - startTime ;
    566      fParticleChange.ProposeGlobalTime( fCandidateEndGlobalTime ) ;
    567   }
    568 
    569 
    570   // Now Correct by Lorentz factor to get delta "proper" Time
    571  
    572   G4double  restMass       = track.GetDynamicParticle()->GetMass() ;
    573   G4double deltaProperTime = deltaTime*( restMass/track.GetTotalEnergy() ) ;


::

    (lldb) p track
    (const G4Track) $14 = {
      fCurrentStepNumber = 1
      fPosition = (dx = 0, dy = 0, dz = 0)
      fGlobalTime = 0.10000000149011612
      fLocalTime = 0
      fTrackLength = 0
      fParentID = 0
      fTrackID = 9999
      fVelocity = 299.79245800000001
      fpTouchable = {
        fObj = 0x000000012788ed70
      }
      fpNextTouchable = {
        fObj = 0x000000012788ed70
      }
      fpOriginTouchable = {
        fObj = 0x000000012788ed70
      }
      fpDynamicParticle = 0x000000012788d8f0
      fTrackStatus = fAlive
      fBelowThreshold = false
      fGoodForTracking = false
      fStepLength = 2995
      fWeight = 1
      fpStep = 0x00000001100c60b0
      fVtxPosition = (dx = 0, dy = 0, dz = 0)
      fVtxMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fVtxKineticEnergy = 0.0000028833531986511571
      fpLVAtVertex = 0x00000001101058c0
      fpCreatorProcess = 0x0000000000000000
      fCreatorModelIndex = -1
      fpUserInformation = 0x0000000000000000
      prev_mat = 0x0000000110104b20
      groupvel = 0x0000000110105760
      prev_velocity = 194.51919555664063
      prev_momentum = 0.0000028833531986511571
      is_OpticalPhoton = true
      useGivenVelocity = false
      fpAuxiliaryTrackInformationMap = 0x0000000000000000
    }
    (lldb) p fTransportEndPosition
    (G4ThreeVector) $15 = (dx = 2995, dy = 0, dz = 0)
    (lldb) 


    (lldb) p *stepData.GetPreStepPoint()
    (G4StepPoint) $19 = {
      fPosition = (dx = 0, dy = 0, dz = 0)
      fGlobalTime = 0.10000000149011612
      fLocalTime = 0
      fProperTime = 0
      fMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fKineticEnergy = 0.0000028833531986511571
      fVelocity = 194.51919555664063
      fpTouchable = {
        fObj = 0x000000012788ed70
      }
      fpMaterial = 0x0000000110104b20
      fpMaterialCutsCouple = 0x0000000110609570
      fpSensitiveDetector = 0x0000000000000000
      fSafety = 0
      fPolarization = (dx = 0, dy = 1, dz = 0)
      fStepStatus = fUndefined
      fpProcessDefinedStep = 0x0000000000000000
      fMass = 0
      fCharge = 0
      fMagneticMoment = 0
      fWeight = 1
    }

    (lldb) p fCandidateEndGlobalTime 
    (G4double) $20 = 15.49693803338686


    (lldb) p 2995./194.51919555664063
    (double) $21 = 15.396938031896743
    (lldb) p 2995./194.51919555664063 + 0.10000000149011612
    (double) $22 = 15.49693803338686
    (lldb) 


    (lldb) p track
    (const G4Track) $23 = {
      fCurrentStepNumber = 2
      fPosition = (dx = 2995, dy = 0, dz = 0)
      fGlobalTime = 15.49693803338686
      fLocalTime = 15.396938031896743
      fTrackLength = 2995     
      /// not including current step

      fParentID = 0
      fTrackID = 9999
      fVelocity = 194.51919555664063

      /// this velocity is not for Acrylic ??
      ///
      fpTouchable = {
        fObj = 0x000000012788ed80
      }
      fpNextTouchable = {
        fObj = 0x000000012788ed80
      }
      fpOriginTouchable = {
        fObj = 0x000000012788ed70
      }
      fpDynamicParticle = 0x000000012788d8f0
      fTrackStatus = fAlive
      fBelowThreshold = false
      fGoodForTracking = false
      fStepLength = 10
      fWeight = 1
      fpStep = 0x00000001100c60b0
      fVtxPosition = (dx = 0, dy = 0, dz = 0)
      fVtxMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fVtxKineticEnergy = 0.0000028833531986511571
      fpLVAtVertex = 0x00000001101058c0
      fpCreatorProcess = 0x0000000000000000
      fCreatorModelIndex = -1
      fpUserInformation = 0x0000000000000000
      prev_mat = 0x0000000110104b20
      groupvel = 0x0000000110105760
      prev_velocity = 194.51919555664063
      prev_momentum = 0.0000028833531986511571
      is_OpticalPhoton = true
      useGivenVelocity = false
      fpAuxiliaryTrackInformationMap = 0x0000000000000000
    }


::

    (ldb) p *stepData.GetPreStepPoint()
    (G4StepPoint) $24 = {
      fPosition = (dx = 2995, dy = 0, dz = 0)
      fGlobalTime = 15.49693803338686
      fLocalTime = 15.396938031896743
      fProperTime = 0
      fMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fKineticEnergy = 0.0000028833531986511571
      fVelocity = 194.51919555664063
      fpTouchable = {
        fObj = 0x000000012788ed80
      }
      fpMaterial = 0x00000001100f93d0
      fpMaterialCutsCouple = 0x0000000110608660
      fpSensitiveDetector = 0x0000000000000000
      fSafety = 0.00000000050000000000000003
      fPolarization = (dx = 0, dy = 1, dz = 0)
      fStepStatus = fGeomBoundary
      fpProcessDefinedStep = 0x000000011011f4b0
      fMass = 0
      fCharge = 0
      fMagneticMoment = 0
      fWeight = 1
    }


    (lldb) p stepData.GetPreStepPoint()->GetMaterial()->GetName()
    (const G4String) $26 = (std::__1::string = "Acrylic")
    (lldb) p stepData.GetPostStepPoint()->GetMaterial()->GetName()
    (const G4String) $27 = (std::__1::string = "Acrylic")

    (lldb) p track
    (const G4Track) $28 = {
      fCurrentStepNumber = 3
      fPosition = (dx = 3005, dy = 0, dz = 0)
      fGlobalTime = 15.548346841506715
      fLocalTime = 15.448346840016599
      fTrackLength = 3005
      fParentID = 0
      fTrackID = 9999
      fVelocity = 192.77955627441406
      fpTouchable = {
        fObj = 0x000000012788ed90
      }
      fpNextTouchable = {
        fObj = 0x000000012788ed90
      }
      fpOriginTouchable = {
        fObj = 0x000000012788ed70
      }
      fpDynamicParticle = 0x000000012788d8f0
      fTrackStatus = fAlive
      fBelowThreshold = false
      fGoodForTracking = false
      fStepLength = 990
      fWeight = 1
      fpStep = 0x00000001100c60b0
      fVtxPosition = (dx = 0, dy = 0, dz = 0)
      fVtxMomentumDirection = (dx = 1, dy = 0, dz = 0)
      fVtxKineticEnergy = 0.0000028833531986511571
      fpLVAtVertex = 0x00000001101058c0
      fpCreatorProcess = 0x0000000000000000
      fCreatorModelIndex = -1
      fpUserInformation = 0x0000000000000000
      prev_mat = 0x00000001100f93d0
      groupvel = 0x00000001101004f0
      prev_velocity = 192.77955627441406
      prev_momentum = 0.0000028833531986511571
      is_OpticalPhoton = true
      useGivenVelocity = false
      fpAuxiliaryTrackInformationMap = 0x0000000000000000
    }



::

    (lldb) b "G4Track::SetGlobalTime"

    (lldb) bt
    * thread #1: tid = 0x1059dc, 0x0000000104c76b60 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs() [inlined] G4Track::SetGlobalTime(this=0x000000012818a7b0, aValue=<unavailable>) at G4Track.icc:100, queue = 'com.apple.main-thread', stop reason = breakpoint 4.3
      * frame #0: 0x0000000104c76b60 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs() [inlined] G4Track::SetGlobalTime(this=0x000000012818a7b0, aValue=<unavailable>) at G4Track.icc:100
        frame #1: 0x0000000104c76b60 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs() [inlined] G4Step::UpdateTrack(this=0x000000011127a5f0) + 34 at G4Step.icc:231
        frame #2: 0x0000000104c76b3e libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x000000011127a460) + 510 at G4SteppingManager2.cc:471
        frame #3: 0x0000000104c74771 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000011127a460) + 417 at G4SteppingManager.cc:191
        frame #4: 0x0000000104c7e771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000011127a420, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #5: 0x0000000104bd6727 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000011127a390, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #6: 0x0000000104b58611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010f66ef00, i_event=0) + 49 at G4RunManager.cc:399
        frame #7: 0x0000000104b584db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010f66ef00, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #8: 0x0000000104b57913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010f66ef00, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #9: 0x0000000103ee4882 libcfg4.dylib`CG4::propagate(this=0x000000010f66ee50) + 1458 at CG4.cc:270
        frame #10: 0x0000000103fcd52a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfe3f0) + 538 at OKG4Mgr.cc:82
        frame #11: 0x00000001000139ca OKG4Test`main(argc=29, argv=0x00007fff5fbfe4d0) + 1498 at OKG4Test.cc:57
        frame #12: 0x00007fff915315fd libdyld.dylib`start + 1
    (lldb) 
















::

     49  G4double G4ParticleChange::GetVelocity() const
     50 {
     51    return theVelocityChange;
     52 }
     53 
     54 inline
     55   void G4ParticleChange::ProposeVelocity(G4double finalVelocity)
     56 {
     57    theVelocityChange = finalVelocity;
     58    isVelocityChanged = true;
     59 }
     60 

::

    228 void G4ParticleChange::Initialize(const G4Track& track)
    229 {
    230   // use base class's method at first
    231   G4VParticleChange::Initialize(track);
    232   theCurrentTrack= &track;
    233 
    234   // set Energy/Momentum etc. equal to those of the parent particle
    235   const G4DynamicParticle*  pParticle = track.GetDynamicParticle();
    236   theEnergyChange            = pParticle->GetKineticEnergy();
    237   theVelocityChange          = track.GetVelocity();
    238   isVelocityChanged          = false;
    239   theMomentumDirectionChange = pParticle->GetMomentumDirection();
    240   thePolarizationChange      = pParticle->GetPolarization();
    241   theProperTimeChange        = pParticle->GetProperTime();
    242 
    243   // Set mass/charge/MagneticMoment  of DynamicParticle
    244   theMassChange = pParticle->GetMass();
    245   theChargeChange = pParticle->GetCharge();
    246   theMagneticMomentChange = pParticle->GetMagneticMoment();
    247 
    248   // set Position  equal to those of the parent track
    249   thePositionChange      = track.GetPosition();
    250 
    251   // set TimeChange equal to local time of the parent track
    252   theTimeChange                = track.GetLocalTime();
    253 
    254   // set initial Local/Global time of the parent track
    255   theLocalTime0           = track.GetLocalTime();
    256   theGlobalTime0          = track.GetGlobalTime();
    257 
    258 }


::

    348 G4Step* G4ParticleChange::UpdateStepForPostStep(G4Step* pStep)
    349 {
    350   // A physics process always calculates the final state of the particle
    351 
    352   // Take note that the return type of GetMomentumChange is a
    353   // pointer to G4ParticleMometum. Also it is a normalized 
    354   // momentum vector.
    355 
    356   G4StepPoint* pPostStepPoint = pStep->GetPostStepPoint();
    357   G4Track* pTrack = pStep->GetTrack();
    358 
    359   // Set Mass/Charge
    360   pPostStepPoint->SetMass(theMassChange);
    361   pPostStepPoint->SetCharge(theChargeChange);
    362   pPostStepPoint->SetMagneticMoment(theMagneticMomentChange);
    363 
    364   // update kinetic energy and momentum direction
    365   pPostStepPoint->SetMomentumDirection(theMomentumDirectionChange);
    366   pPostStepPoint->SetKineticEnergy( theEnergyChange );
    367 
    368   // calculate velocity
    369   pTrack->SetKineticEnergy( theEnergyChange );
    370   if (!isVelocityChanged) {
    371     if(theEnergyChange > 0.0) {
    372       theVelocityChange = pTrack->CalculateVelocity();
    373     } else if(theMassChange > 0.0) {
    374       theVelocityChange = 0.0;
    375     }
    376   }
    377   pPostStepPoint->SetVelocity(theVelocityChange);

    ///   the G4ParticleChange::GetVelocity is never called
    ///   so passing on to post is the only place the info
    ///   goes


    378 
    379    // update polarization
    380   pPostStepPoint->SetPolarization( thePolarizationChange );
    381 
    382   // update position and time
    383   pPostStepPoint->SetPosition( thePositionChange  );
    384   pPostStepPoint->AddGlobalTime(theTimeChange - theLocalTime0);
    385   pPostStepPoint->SetLocalTime( theTimeChange );
    386   pPostStepPoint->SetProperTime( theProperTimeChange  );
    387 
    388   if (isParentWeightProposed ){
    389     pPostStepPoint->SetWeight( theParentWeight );
    390   }
    391 
    392 #ifdef G4VERBOSE
    393   G4Track*     aTrack  = pStep->GetTrack();
    394   if (debugFlag) CheckIt(*aTrack);
    395 #endif
    396 
    397   //  Update the G4Step specific attributes 
    398   return UpdateStepInfo(pStep);
    399 }


::

    delta:cfg4 blyth$ g4-cc UpdateStepForPostStep
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/biasing/generic/src/G4ParticleChangeForOccurenceBiasing.cc:G4Step* G4ParticleChangeForOccurenceBiasing::UpdateStepForPostStep(G4Step* step)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/biasing/generic/src/G4ParticleChangeForOccurenceBiasing.cc:  fWrappedParticleChange->UpdateStepForPostStep(step);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITReactionChange.cc:  fParticleChange[stepA->GetTrack()]->UpdateStepForPostStep(stepA);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITReactionChange.cc:  fParticleChange[stepB->GetTrack()]->UpdateStepForPostStep(stepB);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:  fpParticleChange->UpdateStepForPostStep(fpStep);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/parameterisation/src/G4FastStep.cc:G4Step* G4FastStep::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChange.cc:G4Step* G4ParticleChange::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForDecay.cc:G4Step* G4ParticleChangeForDecay::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForGamma.cc:G4Step* G4ParticleChangeForGamma::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForLoss.cc:G4Step* G4ParticleChangeForLoss::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForMSC.cc:G4Step* G4ParticleChangeForMSC::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForTransport.cc:G4Step* G4ParticleChangeForTransport::UpdateStepForPostStep(G4Step* pStep)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForTransport.cc:  // return G4ParticleChange::UpdateStepForPostStep(pStep);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4VParticleChange.cc:G4Step* G4VParticleChange::UpdateStepForPostStep(G4Step* Step)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/tracking/src/G4SteppingManager2.cc:  fParticleChange->UpdateStepForPostStep(fStep);
    delta:cfg4 blyth$ 

::

    526 void G4SteppingManager::InvokePSDIP(size_t np)
    527 {
    528          fCurrentProcess = (*fPostStepDoItVector)[np];
    529          fParticleChange
    530             = fCurrentProcess->PostStepDoIt( *fTrack, *fStep);
    531 
    532          // Update PostStepPoint of Step according to ParticleChange
    533      fParticleChange->UpdateStepForPostStep(fStep);
    ...
    538          // Update G4Track according to ParticleChange after each PostStepDoIt
    539          fStep->UpdateTrack();
    540 
    541          // Update safety after each invocation of PostStepDoIts
    542          fStep->GetPostStepPoint()->SetSafety( CalculateSafety() );
    543 
    544          // Now Store the secondaries from ParticleChange to SecondaryList
    545          G4Track* tempSecondaryTrack;
    546          G4int    num2ndaries;
    547 
    548          num2ndaries = fParticleChange->GetNumberOfSecondaries();
    ...
    ...      skipped 2ndary loop
    ...
    581          // Set the track status according to what the process defined
    582          fTrack->SetTrackStatus( fParticleChange->GetTrackStatus() );
    ...
    585          fParticleChange->Clear();
    586 }



::

    116 G4StepStatus G4SteppingManager::Stepping()
    117 //////////////////////////////////////////
    118 {
    ...
    133 
    134 // Store last PostStepPoint to PreStepPoint, and swap current and nex
    135 // volume information of G4Track. Reset total energy deposit in one Step. 
    136    fStep->CopyPostToPreStepPoint();
    137    fStep->ResetTotalEnergyDeposit();
    138 
    139 // Switch next touchable in track to current one
    140    fTrack->SetTouchableHandle(fTrack->GetNextTouchableHandle());
    ...
    147 //JA Set the volume before it is used (in DefineStepLength() for User Limit) 
    148    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
    149 
    150 // Reset the step's auxiliary points vector pointer
    151    fStep->SetPointerToVectorOfAuxiliaryPoints(0);
    152 
    153 //-----------------
    154 // AtRest Processes
    155 //-----------------
    156 
    157    if( fTrack->GetTrackStatus() == fStopButAlive ){
    158      if( MAXofAtRestLoops>0 ){
    159         InvokeAtRestDoItProcs();
    160         fStepStatus = fAtRestDoItProc;
    161         fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
    162 
    163 #ifdef G4VERBOSE
    164             // !!!!! Verbose
    165              if(verboseLevel>0) fVerbose->AtRestDoItInvoked();
    166 #endif
    167 
    168      }
    169      // Make sure the track is killed
    170      fTrack->SetTrackStatus( fStopAndKill );
    171    }
    172
    173 //---------------------------------
    174 // AlongStep and PostStep Processes
    175 //---------------------------------
    176 
    177 
    178    else{
    179      // Find minimum Step length demanded by active disc./cont. processes
    180      DefinePhysicalStepLength();
    181 
    182      // Store the Step length (geometrical length) to G4Step and G4Track
    183      fStep->SetStepLength( PhysicalStep );
    184      fTrack->SetStepLength( PhysicalStep );
    185      G4double GeomStepLength = PhysicalStep;
    186 
    187      // Store StepStatus to PostStepPoint
    188      fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
    189 
    190      // Invoke AlongStepDoIt 
    191      InvokeAlongStepDoItProcs();
    192 
    193      // Update track by taking into account all changes by AlongStepDoIt
    194      fStep->UpdateTrack();
    195 
    196      // Update safety after invocation of all AlongStepDoIts
    197      endpointSafOrigin= fPostStepPoint->GetPosition();
    198 //     endpointSafety=  std::max( proposedSafety - GeomStepLength, 0.);
    199      endpointSafety=  std::max( proposedSafety - GeomStepLength, kCarTolerance);
    200 
    201      fStep->GetPostStepPoint()->SetSafety( endpointSafety );
    202 
    203 #ifdef G4VERBOSE
    204                          // !!!!! Verbose
    205            if(verboseLevel>0) fVerbose->AlongStepDoItAllDone();
    206 #endif
    207 
    208      // Invoke PostStepDoIt
    209      InvokePostStepDoItProcs();
    210 
    211 #ifdef G4VERBOSE
    212                  // !!!!! Verbose
    213      if(verboseLevel>0) fVerbose->PostStepDoItAllDone();
    214 #endif
    215    }






