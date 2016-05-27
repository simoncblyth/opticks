Optical Local Time Goes Backward
===================================

Presumably some bad material props ? 

* looks to be using spline with ABSLENGTH resulting in crazy interpolation results 


::

    ggv-;ggv-g4gun --dbg

::

      G4ParticleChange::CheckIt    : the local time goes back  !!  Difference:  6.9082[ns] 
      opticalphoton E=3.19939e-06 pos=-18.0795, -799.699, -6.59899 global time=11.7134 local time=0 proper time=0
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :            -1.29e+03      <<<<<<<<<<<<<<<<<<<<<<<<  NEGATIVE STEP LENGTH <<<<<<<<<<<<<<<<<<<<<
            Stepping Control      :                    0


Position in meter::

    (lldb) 
       554        G4cout << aTrack.GetDefinition()->GetParticleName()
       555           << " E=" << aTrack.GetKineticEnergy()/MeV
       556           << " pos=" << aTrack.GetPosition().x()/m
       557           << ", " << aTrack.GetPosition().y()/m
       558           << ", " << aTrack.GetPosition().z()/m
       559           << " global time=" << aTrack.GetGlobalTime()/ns
       560           << " local time=" << aTrack.GetLocalTime()/ns



::

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : TRACK003
          issued by : G4ParticleChange::CheckIt
    momentum, energy, and/or time was illegal
    *** Event Must Be Aborted ***
    -------- EEEE -------- G4Exception-END --------- EEEE -------

      G4VParticleChange::CheckIt    : the true step length is negative  !!  Difference:  1289.99[MeV] 
    opticalphoton E=3.19939e-06 pos=-18.6742, -800.248, -5.59423
          -----------------------------------------------
            G4ParticleChange Information  


The CheckIt is after the problem happens, what sets the debug flag?::

    (lldb) b "G4ParticleChange::CheckIt(G4Track const&)" 


    (lldb) bt
    * thread #1: tid = 0x71d4a0, 0x0000000105ed6e60 libG4track.dylib`G4ParticleChange::CheckIt(this=0x00000001092886c0, aTrack=0x000000010e3338c0) + 32 at G4ParticleChange.cc:508, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000105ed6e60 libG4track.dylib`G4ParticleChange::CheckIt(this=0x00000001092886c0, aTrack=0x000000010e3338c0) + 32 at G4ParticleChange.cc:508
        frame #1: 0x0000000105ee063f libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(this=0x00000001092886c0, pStep=0x000000010910e410) + 1519 at G4ParticleChangeForTransport.cc:202
        frame #2: 0x0000000102ecf96e libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x000000010910e280) + 254 at G4SteppingManager2.cc:420
        frame #3: 0x0000000102ecb168 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010910e280) + 504 at G4SteppingManager.cc:191
        frame #4: 0x0000000102ee292d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010910e240, apValueG4Track=0x000000010e3338c0) + 1357 at G4TrackingManager.cc:126
        frame #5: 0x0000000102dbfe44 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010910e1b0, anEvent=0x000000010e3323d0) + 3188 at G4EventManager.cc:185
        frame #6: 0x0000000102dc0b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010910e1b0, anEvent=0x000000010e3323d0) + 47 at G4EventManager.cc:336
        frame #7: 0x0000000102cedc75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000109003060, i_event=0) + 69 at G4RunManager.cc:399
        frame #8: 0x0000000102cedab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #9: 0x0000000102cec8e4 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #10: 0x000000010155b38d libcfg4.dylib`CG4::propagate(this=0x0000000108721880) + 605 at CG4.cc:181
        frame #11: 0x000000010000d542 CG4Test`main(argc=16, argv=0x00007fff5fbfdca0) + 210 at CG4Test.cc:20
        frame #12: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 


Who sets debugFlag ? The setter is optimized out of the binary, so never called::

    (lldb) f 0
    frame #0: 0x0000000105eeee60 libG4track.dylib`G4ParticleChange::CheckIt(this=0x000000010d0dc8f0, aTrack=0x000000010eaa4e30) + 32 at G4ParticleChange.cc:508
       505  
       506  G4bool G4ParticleChange::CheckIt(const G4Track& aTrack)
       507  {
    -> 508    G4bool    exitWithError = false;
       509    G4double  accuracy;
       510    static G4ThreadLocal G4int nError = 0;
       511  #ifdef G4VERBOSE
    (lldb) f 1
    frame #1: 0x0000000105ef863f libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(this=0x000000010d0dc8f0, pStep=0x000000010910e410) + 1519 at G4ParticleChangeForTransport.cc:202
       199                   - pPreStepPoint->GetProperTime());
       200  
       201  #ifdef G4VERBOSE
    -> 202    if (debugFlag) CheckIt(*aTrack);
       203  #endif
       204  
       205    //  Update the G4Step specific attributes
    (lldb) 


::

    simon:cfg4 blyth$ g4-icc debugFlag
    /usr/local/env/g4/geant4.10.02/source/track/include/G4VParticleChange.icc:  debugFlag = false;
    /usr/local/env/g4/geant4.10.02/source/track/include/G4VParticleChange.icc:  debugFlag = true;
    /usr/local/env/g4/geant4.10.02/source/track/include/G4VParticleChange.icc:  return debugFlag;

    289 inline
    290  void G4VParticleChange::ClearDebugFlag()
    291 {
    292   debugFlag = false;
    293 }
    294 
    295 inline
    296  void G4VParticleChange::SetDebugFlag()
    297 {
    298   debugFlag = true;
    299 }
    300 
    301 inline
    302  G4bool G4VParticleChange::GetDebugFlag() const
    303 {
    304   return debugFlag;
    305 }



::

    simon:ggeo blyth$ g4-cc debugFlag
    /usr/local/env/g4/geant4.10.02/source/processes/parameterisation/src/G4FastStep.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/processes/parameterisation/src/G4FastStep.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChange.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChange.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChange.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForDecay.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForGamma.cc:  debugFlag = false;
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForLoss.cc:  debugFlag = false;
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForTransport.cc:  if (debugFlag) CheckIt(*aTrack);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:   debugFlag(false)
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:  debugFlag = true;
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:   debugFlag(right.debugFlag)
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:    debugFlag = right.debugFlag;
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:  if (debugFlag) CheckSecondary(*aTrack);
    simon:ggeo blyth$ 



Following removal of physics retreive the problem now seems repeatable::

    [2016-May-27 17:56:24.186048]:info: CSA tots evt 1 trk 10 stp 44
    [2016-May-27 17:56:24.186155]:info: CSA tots evt 1 trk 11 stp 45
    [2016-May-27 17:56:24.186261]:info: CSA tots evt 1 trk 11 stp 46
    [2016-May-27 17:56:24.186371]:info: CSA tots evt 1 trk 11 stp 47
    [2016-May-27 17:56:24.186476]:info: CSA tots evt 1 trk 11 stp 48
    [2016-May-27 17:56:24.186582]:info: CSA tots evt 1 trk 11 stp 49
    [2016-May-27 17:56:24.186691]:info: CSA tots evt 1 trk 11 stp 50
    [2016-May-27 17:56:24.186786]:info: CSA tots evt 1 trk 11 stp 51
      G4ParticleChange::CheckIt    : the local time goes back  !!  Difference:  8.46245[ns] 
    opticalphoton E=3.13437e-06 pos=-18.0795, -799.699, -6.59913 global time=1.72415 local time=0 proper time=0
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :            -1.59e+03
            Stepping Control      :                    0



Not flaky anymore::

    [2016-May-27 17:57:45.214452]:info: CSA tots evt 1 trk 11 stp 49
    [2016-May-27 17:57:45.214564]:info: CSA tots evt 1 trk 11 stp 50
    [2016-May-27 17:57:45.214659]:info: CSA tots evt 1 trk 11 stp 51
      G4ParticleChange::CheckIt    : the local time goes back  !!  Difference:  8.46245[ns] 
    opticalphoton E=3.13437e-06 pos=-18.0795, -799.699, -6.59913 global time=1.72415 local time=0 proper time=0
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :            -1.59e+03
            Stepping Control      :                    0
        First Step In the voulme  : 
            Mass (GeV)   :                    0
            Charge (eplus)   :                    0
            MagneticMoment   :                    0
                    :  =                    0*[e hbar]/[2 m]





::

    059 class G4Transportation : public G4VProcess
     60 {
     61   // Concrete class that does the geometrical transport 

    184 
    185      G4ParticleChangeForTransport fParticleChange;
    186        // New ParticleChange


::

    simon:geant4.10.02 blyth$ grep fParticleChange source/processes/transportation/src/G4Transportation.cc
        // fParticleChange(),
      pParticleChange= &fParticleChange;   // Required to conform to G4VProcess 
      fParticleChange.ProposeFirstStepInVolume(fFirstStepInVolume);
      fParticleChange.ProposeTrueStepLength(geometryStepLength) ;
      fParticleChange.Initialize(track) ;
      fParticleChange.ProposePosition(fTransportEndPosition) ;
      fParticleChange.ProposeMomentumDirection(fTransportEndMomentumDir) ;
      fParticleChange.ProposeEnergy(fTransportEndKineticEnergy) ;
      fParticleChange.SetMomentumChanged(fMomentumChanged) ;
      fParticleChange.ProposePolarization(fTransportEndSpin);
         fParticleChange.ProposeLocalTime(  track.GetLocalTime() + deltaTime) ;
         fParticleChange.ProposeGlobalTime( fCandidateEndGlobalTime ) ;
      fParticleChange.ProposeProperTime(track.GetProperTime() + deltaProperTime) ;
      //fParticleChange. ProposeTrueStepLength( track.GetStepLength() ) ;
            fParticleChange.ProposeTrackStatus( fStopAndKill )  ;
      fParticleChange.SetPointerToVectorOfAuxiliaryPoints
      return &fParticleChange ;
      // fParticleChange.Initialize(track) ;  // To initialise TouchableChange
      fParticleChange.ProposeTrackStatus(track.GetTrackStatus()) ;
           fParticleChange.ProposeTrackStatus( fStopAndKill ) ;
        fParticleChange.SetTouchableHandle( fCurrentTouchableHandle ) ;
        fParticleChange.SetTouchableHandle( track.GetTouchableHandle() ) ;
      fParticleChange.ProposeFirstStepInVolume(fFirstStepInVolume);
      fParticleChange.ProposeLastStepInVolume(isLastStep);    
      fParticleChange.SetMaterialInTouchable( (G4Material *) pNewMaterial ) ;
      fParticleChange.SetSensitiveDetectorInTouchable( (G4VSensitiveDetector *) pNewSensitiveDetector ) ;
      fParticleChange.SetMaterialCutsCoupleInTouchable( pNewMaterialCutsCouple );
      fParticleChange.SetTouchableHandle(retCurrentTouchable) ;
      return &fParticleChange ;

::

    g4-;g4-cls G4Transportation



Breakpoint in context to watch::

    (lldb) b "G4Transportation::AlongStepDoIt(G4Track const&, G4Step const&)" 

    (lldb) p fParticleChange.debugFlag
    (G4bool) $0 = true

    ## huh it starts true 


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


::

    (lldb) c
    Process 86040 resuming
    Process 86040 stopped
    * thread #1: tid = 0x737689, 0x000000010434be1c libG4processes.dylib`G4Transportation::AlongStepDoIt(this=0x000000010db00140, track=0x000000010eb97860, stepData=0x000000010910e410) + 428 at G4Transportation.cc:561, queue = 'com.apple.main-thread', stop reason = breakpoint 3.1
        frame #0: 0x000000010434be1c libG4processes.dylib`G4Transportation::AlongStepDoIt(this=0x000000010db00140, track=0x000000010eb97860, stepData=0x000000010910e410) + 428 at G4Transportation.cc:561
       558       if ( initialVelocity > 0.0 )  { deltaTime = stepLength/initialVelocity; }
       559  
       560       fCandidateEndGlobalTime   = startTime + deltaTime ;
    -> 561       fParticleChange.ProposeLocalTime(  track.GetLocalTime() + deltaTime) ;
       562    }
       563    else
       564    {
    (lldb) p fEndGlobalTimeComputed
    (G4bool) $7 = false
    (lldb) p initialVelocity
    (G4double) $8 = 299.78858073278644
    (lldb) p stepLength
    (G4double) $9 = 2.1233660778606285
    (lldb) p startTime
    (G4double) $10 = 0
    (lldb) p deltaTime
    (G4double) $11 = 0.0070828784494405732



Conditional breakpoint hit just before::

    (lldb) b G4Transportation.cc:561
    Breakpoint 1: where = libG4processes.dylib`G4Transportation::AlongStepDoIt(G4Track const&, G4Step const&) + 428 at G4Transportation.cc:561, address = 0x000000010434be1c

    rocess 86767 stopped
    * thread #1: tid = 0x7382af, 0x000000010434be1c libG4processes.dylib`G4Transportation::AlongStepDoIt(this=0x000000010a41a560, track=0x000000010ebe0660, stepData=0x0000000109030310) + 428 at G4Transportation.cc:561, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x000000010434be1c libG4processes.dylib`G4Transportation::AlongStepDoIt(this=0x000000010a41a560, track=0x000000010ebe0660, stepData=0x0000000109030310) + 428 at G4Transportation.cc:561
       558       if ( initialVelocity > 0.0 )  { deltaTime = stepLength/initialVelocity; }
       559  
       560       fCandidateEndGlobalTime   = startTime + deltaTime ;
    -> 561       fParticleChange.ProposeLocalTime(  track.GetLocalTime() + deltaTime) ;
       562    }
       563    else
       564    {
    (lldb) br mod -c 'deltaTime < 0'
    (lldb) 


::

    (lldb) p deltaTime
    (G4double) $51 = -8.4624468654680545
    (lldb) p startTime
    (G4double) $52 = 1.7241540937693964
    (lldb) p initialVelocity
    (G4double) $53 = 188.47366895827508


    ####### negative stepLength the cause of deltaTime negative 
    (lldb) p stepLength
    (G4double) $54 = -1594.9484090992187


    (lldb) p fTransportEndPosition
    (G4ThreeVector) $55 = (dx = -18455.487573077644, dy = -798314.83047917974, dz = -7295.820924435935)
    (lldb) p track
    (const G4Track) $56 = {
      fCurrentStepNumber = 1
      fPosition = (dx = -18079.490838288788, dy = -799699.42585238547, dz = -6599.1269201459945)
      fGlobalTime = 1.7241540937693964
      fLocalTime = 0
      fTrackLength = 0
      fParentID = 1
      fTrackID = 7
      fVelocity = 299.79245800000001
      fpTouchable = {
        fObj = 0x000000010ebe1620
      }
      fpNextTouchable = {
        fObj = 0x000000010ebe1620
      }
      fpOriginTouchable = {
        fObj = 0x000000010ebe1620
      }
      fpDynamicParticle = 0x000000010ebe04d0
      fTrackStatus = fAlive
      fBelowThreshold = false
      fGoodForTracking = false
      fStepLength = -1594.9484090992187
      fWeight = 1
      fpStep = 0x0000000109030310
      fVtxPosition = (dx = -18079.490838288788, dy = -799699.42585238547, dz = -6599.1269201459945)
      fVtxMomentumDirection = (dx = 0.2357422551373986, dy = -0.86811295293726964, dz = 0.4368128776550293)
      fVtxKineticEnergy = 0.0000031343675198922904
      fpLVAtVertex = 0x000000010917bdf0
      fpCreatorProcess = 0x000000010a441430
      fCreatorModelIndex = -1
      fpUserInformation = 0x0000000000000000
      prev_mat = 0x0000000109116ad0
      groupvel = 0x000000010ec005f0
      prev_velocity = 188.47366895827508
      prev_momentum = 0.0000031343675198922904
      is_OpticalPhoton = true
      useGivenVelocity = false
      fpAuxiliaryTrackInformationMap = 0x0000000000000000
    }
    (lldb) p track.fpCreatorProcess
    (const G4VProcess *const) $57 = 0x000000010a441430

    (lldb) p *(track.fpCreatorProcess)
    (const G4VProcess) $58 = {}

    ## maybe no members listed as no natives 

    (lldb) p track.fpCreatorProcess->theProcessName
    (const G4String) $62 = (std::__1::string = "Scintillation")

    (lldb) p *stepData.fpPreStepPoint->fpMaterial
    (G4Material) $67 = {
      fName = (std::__1::string = "/dd/Materials/GdDopedLS")

    (lldb) p *stepData.fpPostStepPoint->fpMaterial
    (G4Material) $68 = {
      fName = (std::__1::string = "/dd/Materials/GdDopedLS")

    ## same material instance 

    lldb) p stepData.fpPostStepPoint->fpMaterial
    (G4Material *) $76 = 0x0000000109116ad0
    (lldb) p stepData.fpPreStepPoint->fpMaterial
    (G4Material *) $77 = 0x0000000109116ad0
    (lldb) 


    (lldb) p stepData.fpPreStepPoint->GetVelocity()
    (G4double) $70 = 188.47366895827508
    (lldb) p stepData.fpPostStepPoint->GetVelocity()
    (G4double) $71 = 188.47366895827508


    Huh why this is speed of light but the above are the calulated groupvel  ?

    (lldb) p fParticleChange.theVelocityChange
    (G4double) $73 = 299.79245800000001


    (lldb) p *track.groupvel
    (G4MaterialPropertyVector) $79 = {
      G4PhysicsVector = {
        type = T_G4PhysicsOrderedFreeVector
        edgeMin = 0.0000015120022870975581
        edgeMax = 0.000020664031256999959
        numberOfNodes = 39
        dataVector = size=39 {
          [0] = 202.82284388289375
          [1] = 202.82284388289375
          [2] = 200.94046361977814
          [3] = 200.93107465976249
          [4] = 200.93107736499533
          [5] = 200.93048625837784
          [6] = 200.93106717873997
          [7] = 199.73308402083336
          [8] = 198.73232069232421
          [9] = 198.73284723511136
          [10] = 198.73235358678988
          [11] = 198.73236302731044
          [12] = 198.63019840564712
          [13] = 198.50882034223943
          [14] = 197.50190303348799
          [15] = 195.27491170741507
          [16] = 195.27454576399703
          [17] = 195.47186565073739
          [18] = 195.91478235841396
          [19] = 195.91481452634309
          [20] = 194.74585841550916
          [21] = 192.5107550161905
          [22] = 186.73312391977188
          [23] = 186.73300227037515
          [24] = 186.73340972359088
          [25] = 186.73335117744915
          [26] = 186.73355114727215
          [27] = 166.31829720252409
          [28] = 166.31886737943697
          [29] = 166.31970748749424
          [30] = 166.32069391661918
          [31] = 166.32186856566847
          [32] = 190.60400279386732
          [33] = 76.630847824310763
          [34] = 173.44587345856391
          [35] = 192.29888905546647
          [36] = 206.24138172548928
          [37] = 206.24138172548928
          [38] = 206.24138172548928
        }





G4Scintillation.cc::

    433                 // Generate new G4Track object:
    434 
    435                 G4double rand;
    436 
    437                 if (aParticle->GetDefinition()->GetPDGCharge() != 0) {
    438                    rand = G4UniformRand();
    439                 } else {
    440                    rand = 1.0;
    441                 }
    442 
    443                 G4double delta = rand * aStep.GetStepLength();
    444                 G4double deltaTime = delta / (pPreStepPoint->GetVelocity()+
    445                                       rand*(pPostStepPoint->GetVelocity()-
    446                                             pPreStepPoint->GetVelocity())/2.);
    447 
    448                 // emission time distribution
    449                 if (ScintillationRiseTime==0.0) {
    450                    deltaTime = deltaTime -
    451                           ScintillationTime * std::log( G4UniformRand() );
    452                 } else {
    453                    deltaTime = deltaTime +
    454                           sample_time(ScintillationRiseTime, ScintillationTime);
    455                 }
    456 
    457                 G4double aSecondaryTime = t0 + deltaTime;
    458 
    459                 G4ThreeVector aSecondaryPosition =
    460                                     x0 + rand * aStep.GetDeltaPosition();
    461 
    462                 G4Track* aSecondaryTrack = new G4Track(aScintillationPhoton,
    463                                                        aSecondaryTime,
    464                                                        aSecondaryPosition);
    465 
    466                 aSecondaryTrack->SetTouchableHandle(
    467                                  aStep.GetPreStepPoint()->GetTouchableHandle());
    468                 // aSecondaryTrack->SetTouchableHandle((G4VTouchable*)0);
    469 
    470                 aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    471 
    472                 aParticleChange.AddSecondary(aSecondaryTrack);



G4Track.hh::

    225    G4double GetStepLength() const;
    226    void SetStepLength(G4double value);
    227       // Before the end of the AlongStepDoIt loop,StepLength keeps 
    228       // the initial value which is determined by the shortest geometrical Step 
    229       // proposed by a physics process. After finishing the AlongStepDoIt,
    230       // it will be set equal to 'StepLength' in G4Step.
    231 

G4Step.hh::

    106    // step length
    107    G4double GetStepLength() const;
    108    void SetStepLength(G4double value);
    109     // Before the end of the AlongStepDoIt loop,StepLength keeps
    110     // the initial value which is determined by the shortest geometrical Step
    111     // proposed by a physics process. After finishing the AlongStepDoIt,
    112     // it will be set equal to 'StepLength' in G4Step. 


::

    simon:cfg4 blyth$ g4-cc SetStepLength 
    /usr/local/env/g4/geant4.10.02/source/processes/biasing/importance/src/G4ImportanceProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/env/g4/geant4.10.02/source/processes/biasing/importance/src/G4WeightCutOffProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/env/g4/geant4.10.02/source/processes/biasing/importance/src/G4WeightWindowProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/env/g4/geant4.10.02/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:    fpTrack->SetStepLength(fpState->fPhysicalStep);
    /usr/local/env/g4/geant4.10.02/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:    fpStep->SetStepLength(fpState->fPhysicalStep);
    /usr/local/env/g4/geant4.10.02/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:  fpStep->SetStepLength(0.);  //the particle has stopped
    /usr/local/env/g4/geant4.10.02/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:  fpTrack->SetStepLength(0.);
    /usr/local/env/g4/geant4.10.02/source/processes/scoring/src/G4ParallelWorldProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/env/g4/geant4.10.02/source/processes/scoring/src/G4ParallelWorldProcess.cc:    fpHyperStep->SetStepLength(step.GetStepLength());
    /usr/local/env/g4/geant4.10.02/source/processes/scoring/src/G4ParallelWorldScoringProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/env/g4/geant4.10.02/source/processes/scoring/src/G4ScoreSplittingProcess.cc:        fSplitStep->SetStepLength(stepLength);
    /usr/local/env/g4/geant4.10.02/source/processes/scoring/src/G4ScoreSplittingProcess.cc:  fSplitStep->SetStepLength(step.GetStepLength());
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForGamma.cc:  pStep->SetStepLength( 0.0 );
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForMSC.cc:  pStep->SetStepLength(theTrueStepLength);
    /usr/local/env/g4/geant4.10.02/source/track/src/G4ParticleChangeForTransport.cc:  //pStep->SetStepLength( theTrueStepLength );
    /usr/local/env/g4/geant4.10.02/source/track/src/G4VParticleChange.cc:  pStep->SetStepLength( theTrueStepLength );
    /usr/local/env/g4/geant4.10.02/source/tracking/src/G4SteppingManager.cc:     fStep->SetStepLength( PhysicalStep );
    /usr/local/env/g4/geant4.10.02/source/tracking/src/G4SteppingManager.cc:     fTrack->SetStepLength( PhysicalStep );
    /usr/local/env/g4/geant4.10.02/source/tracking/src/G4SteppingManager2.cc:   fStep->SetStepLength( 0. );  //the particle has stopped
    /usr/local/env/g4/geant4.10.02/source/tracking/src/G4SteppingManager2.cc:   fTrack->SetStepLength( 0. );
    /usr/local/env/g4/geant4.10.02/source/visualization/RayTracer/src/G4RayTrajectory.cc:  trajectoryPoint->SetStepLength(aStep->GetStepLength());
    simon:cfg4 blyth$ 

::

    202 G4Step* G4VParticleChange::UpdateStepInfo(G4Step* pStep)
    203 {
    204   // Update the G4Step specific attributes
    205   pStep->SetStepLength( theTrueStepLength );
    206   pStep->AddTotalEnergyDeposit( theLocalEnergyDeposit );
    207   pStep->AddNonIonizingEnergyDeposit( theNonIonizingEnergyDeposit );
    208   pStep->SetControlFlag( theSteppingControlFlag );
    209 
    210   if (theFirstStepInVolume) {pStep->SetFirstStepFlag();}
    211   else                      {pStep->ClearFirstStepFlag();}
    212   if (theLastStepInVolume)  {pStep->SetLastStepFlag();}
    213   else                      {pStep->ClearLastStepFlag();}
    214 
    215   return pStep;
    216 }

::

    252 void G4VParticleChange::DumpInfo() const
    253 {
    254 
    255 // Show header
    256   G4int olprc = G4cout.precision(3);
    257   G4cout << "      -----------------------------------------------"
    258        << G4endl;
    259   G4cout << "        G4ParticleChange Information  " << std::setw(20) << G4endl;
    260   G4cout << "      -----------------------------------------------"
    261        << G4endl;
    ...
    301   G4cout << "        True Path Length (mm) : "
    302        << std::setw(20) << theTrueStepLength/mm
    303        << G4endl;
    304   G4cout << "        Stepping Control      : "
    305        << std::setw(20) << theSteppingControlFlag
    306        << G4endl;





::

    (lldb) bt
    * thread #1: tid = 0x740adf, 0x0000000105eeee60 libG4track.dylib`G4ParticleChange::CheckIt(this=0x000000010d3012a0, aTrack=0x00000001094da2a0) + 32 at G4ParticleChange.cc:508, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000105eeee60 libG4track.dylib`G4ParticleChange::CheckIt(this=0x000000010d3012a0, aTrack=0x00000001094da2a0) + 32 at G4ParticleChange.cc:508
        frame #1: 0x0000000105ef863f libG4track.dylib`G4ParticleChangeForTransport::UpdateStepForAlongStep(this=0x000000010d3012a0, pStep=0x000000010910e410) + 1519 at G4ParticleChangeForTransport.cc:202
        frame #2: 0x0000000102ee796e libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x000000010910e280) + 254 at G4SteppingManager2.cc:420
        frame #3: 0x0000000102ee3168 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010910e280) + 504 at G4SteppingManager.cc:191
        frame #4: 0x0000000102efa92d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010910e240, apValueG4Track=0x00000001094da2a0) + 1357 at G4TrackingManager.cc:126
        frame #5: 0x0000000102dd7e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010910e1b0, anEvent=0x00000001094d8db0) + 3188 at G4EventManager.cc:185
        frame #6: 0x0000000102dd8b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010910e1b0, anEvent=0x00000001094d8db0) + 47 at G4EventManager.cc:336
        frame #7: 0x0000000102d05c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000109003060, i_event=0) + 69 at G4RunManager.cc:399
        frame #8: 0x0000000102d05ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #9: 0x0000000102d048e4 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #10: 0x000000010155b35d libcfg4.dylib`CG4::propagate(this=0x0000000108721880) + 605 at CG4.cc:180
        frame #11: 0x000000010000d542 CG4Test`main(argc=16, argv=0x00007fff5fbfd668) + 210 at CG4Test.cc:20
        frame #12: 0x00007fff89e755fd libdyld.dylib`start + 1
        frame #13: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x0000000102efa92d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010910e240, apValueG4Track=0x00000001094da2a0) + 1357 at G4TrackingManager.cc:126
       123           (fpTrack->GetTrackStatus() == fStopButAlive) ){
       124  
       125      fpTrack->IncrementCurrentStepNumber();
    -> 126      fpSteppingManager->Stepping();
       127  #ifdef G4_STORE_TRAJECTORY
       128      if(StoreTrajectory) fpTrajectory->
       129                          AppendStep(fpSteppingManager->GetStep()); 
    (lldb) f 3
    frame #3: 0x0000000102ee3168 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010910e280) + 504 at G4SteppingManager.cc:191
       188       fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
       189  
       190       // Invoke AlongStepDoIt 
    -> 191       InvokeAlongStepDoItProcs();
       192  
       193       // Update track by taking into account all changes by AlongStepDoIt
       194       fStep->UpdateTrack();
    (lldb) 



    (lldb) f 2
    frame #2: 0x0000000102ee796e libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x000000010910e280) + 254 at G4SteppingManager2.cc:420
       417         = fCurrentProcess->AlongStepDoIt( *fTrack, *fStep );
       418  
       419       // Update the PostStepPoint of Step according to ParticleChange
    -> 420       fParticleChange->UpdateStepForAlongStep(fStep);
       421  #ifdef G4VERBOSE
       422                           // !!!!! Verbose
       423                 if(verboseLevel>0) fVerbose->AlongStepDoItOneByOne();
    (lldb) 



Debug flag starts true and it is never gets cleared so why not swamped with output::

    (lldb) b "G4VParticleChange::ClearDebugFlag()"    ## never hit




Problem with watchpoints is that the objects keep getting created so 
have to fiddle to break at the appropriate point after the object with issue (ie track instance) 
is created and before the issue, in order to know the address to watch. 

Its easier to set conditional breakpoint in a setter::


    (lldb) b "G4Track::SetStepLength(double)" 
    Breakpoint 2: 2 locations.
    (lldb) c
    Process 14140 resuming
    Process 14140 stopped
    * thread #1: tid = 0x744f2c, 0x0000000102ef1321 libG4tracking.dylib`G4Track::SetStepLength(this=0x000000010e547c30, value=2.7644001831851437) + 17 at G4Track.icc:245, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x0000000102ef1321 libG4tracking.dylib`G4Track::SetStepLength(this=0x000000010e547c30, value=2.7644001831851437) + 17 at G4Track.icc:245
       242     { return fStepLength; }
       243  
       244     inline void G4Track::SetStepLength(G4double value)
    -> 245     { fStepLength = value; }
       246  
       247  // vertex (where this track was created) information
       248     inline const G4ThreeVector& G4Track::GetVertexPosition() const
    (lldb) br mod -c 'value < 0'



::

    (lldb) [2016-May-27 20:07:03.515821]:info: CSA tots evt 1 trk 2 stp 2
    [2016-May-27 20:07:03.529331]:info: CSA tots evt 1 trk 2 stp 3
    [2016-May-27 20:07:03.543318]:info: CSA tots evt 1 trk 2 stp 4
    [2016-May-27 20:07:03.556692]:info: CSA tots evt 1 trk 2 stp 5
    [2016-May-27 20:07:03.571641]:info: CSA tots evt 1 trk 2 stp 6
    [2016-May-27 20:07:03.585871]:info: CSA tots evt 1 trk 2 stp 7
    [2016-May-27 20:07:03.600144]:info: CSA tots evt 1 trk 2 stp 8
    Process 14140 stopped
    * thread #1: tid = 0x744f2c, 0x0000000102ef1321 libG4tracking.dylib`G4Track::SetStepLength(this=0x000000010d328780, value=-340.79660414597521) + 17 at G4Track.icc:245, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x0000000102ef1321 libG4tracking.dylib`G4Track::SetStepLength(this=0x000000010d328780, value=-340.79660414597521) + 17 at G4Track.icc:245
       242     { return fStepLength; }
       243  
       244     inline void G4Track::SetStepLength(G4double value)
    -> 245     { fStepLength = value; }
       246  
       247  // vertex (where this track was created) information
       248     inline const G4ThreeVector& G4Track::GetVertexPosition() const
    (lldb) p value
    (G4double) $46 = -340.79660414597521
    (lldb) bt
    * thread #1: tid = 0x744f2c, 0x0000000102ef1321 libG4tracking.dylib`G4Track::SetStepLength(this=0x000000010d328780, value=-340.79660414597521) + 17 at G4Track.icc:245, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
      * frame #0: 0x0000000102ef1321 libG4tracking.dylib`G4Track::SetStepLength(this=0x000000010d328780, value=-340.79660414597521) + 17 at G4Track.icc:245
        frame #1: 0x0000000102ef0139 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010910e280) + 457 at G4SteppingManager.cc:184
        frame #2: 0x0000000102f0792d libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010910e240, apValueG4Track=0x000000010d328780) + 1357 at G4TrackingManager.cc:126
        frame #3: 0x0000000102de4e44 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010910e1b0, anEvent=0x000000010e546740) + 3188 at G4EventManager.cc:185
        frame #4: 0x0000000102de5b2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010910e1b0, anEvent=0x000000010e546740) + 47 at G4EventManager.cc:336
        frame #5: 0x0000000102d12c75 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000109003060, i_event=0) + 69 at G4RunManager.cc:399
        frame #6: 0x0000000102d12ab5 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 101 at G4RunManager.cc:367
        frame #7: 0x0000000102d118e4 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000109003060, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 196 at G4RunManager.cc:273
        frame #8: 0x000000010155b9ed libcfg4.dylib`CG4::propagate(this=0x0000000108721880) + 605 at CG4.cc:180
        frame #9: 0x000000010000d542 CG4Test`main(argc=16, argv=0x00007fff5fbfd598) + 210 at CG4Test.cc:20
        frame #10: 0x00007fff89e755fd libdyld.dylib`start + 1
        frame #11: 0x00007fff89e755fd libdyld.dylib`start + 1
    (lldb) 


::

    (lldb) f 1
    frame #1: 0x0000000102ef0139 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010910e280) + 457 at G4SteppingManager.cc:184
       181  
       182       // Store the Step length (geometrical length) to G4Step and G4Track
       183       fStep->SetStepLength( PhysicalStep );
    -> 184       fTrack->SetStepLength( PhysicalStep );
       185       G4double GeomStepLength = PhysicalStep;
       186  
       187       // Store StepStatus to PostStepPoint

    (lldb) p PhysicalStep
    (G4double) $47 = -340.79660414597521



::

    g4-cls G4SteppingManager

::

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


::

    188    private:
    189 //---------   
    190 
    191 // Member functions
    192 
    193    void DefinePhysicalStepLength();
    194       // Calculate corresponding physical length from the mean free path 
    195       // left for each discrete phyiscs process. The minimum allowable
    196       // Step for each continious process will be also calculated.



::

    simon:geant4.10.02 blyth$ g4-cls G4SteppingManager2
    vi -R source/tracking/src/G4SteppingManager2.cc

    (lldb) b G4SteppingManager::DefinePhysicalStepLength() 


    (lldb) b 181
    Breakpoint 2: where = libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength() + 355 at G4SteppingManager2.cc:181, address = 0x0000000102ef37d3
    (lldb) c
    Process 14976 resuming
    Process 14976 stopped
    * thread #1: tid = 0x745f1d, 0x0000000102ef37d3 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x000000010910e280) + 355 at G4SteppingManager2.cc:181, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x0000000102ef37d3 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x000000010910e280) + 355 at G4SteppingManager2.cc:181
       178       if(verboseLevel>0) fVerbose->DPSLPostStep();
       179  #endif
       180  
    -> 181       switch (fCondition) {
       182       case ExclusivelyForced:
       183           (*fSelectedPostStepDoItVector)[np] = ExclusivelyForced;
       184           fStepStatus = fExclusivelyForcedProc;
    (lldb) br mod -c 'physIntLength < 0'

    lldb) br dis 1
    1 breakpoints disabled.
    (lldb) c
    Process 14976 resuming
    (lldb) [2016-May-27 20:19:24.485726]:info: CSA tots evt 1 trk 2 stp 3
    [2016-May-27 20:19:24.498583]:info: CSA tots evt 1 trk 2 stp 4
    [2016-May-27 20:19:24.512392]:info: CSA tots evt 1 trk 2 stp 5
    [2016-May-27 20:19:24.525881]:info: CSA tots evt 1 trk 2 stp 6
    [2016-May-27 20:19:24.539746]:info: CSA tots evt 1 trk 2 stp 7
    [2016-May-27 20:19:24.552401]:info: CSA tots evt 1 trk 2 stp 8
    Process 14976 stopped
    * thread #1: tid = 0x745f1d, 0x0000000102ef37d3 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x000000010910e280) + 355 at G4SteppingManager2.cc:181, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x0000000102ef37d3 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x000000010910e280) + 355 at G4SteppingManager2.cc:181
       178       if(verboseLevel>0) fVerbose->DPSLPostStep();
       179  #endif
       180  
    -> 181       switch (fCondition) {
       182       case ExclusivelyForced:
       183           (*fSelectedPostStepDoItVector)[np] = ExclusivelyForced;
       184           fStepStatus = fExclusivelyForcedProc;
    (lldb) p physIntLength
    (G4double) $44 = -340.79660414597521

    (lldb) p fCurrentProcess->theProcessName
    (G4String) $49 = (std::__1::string = "OpAbsorption")

::

    (lldb) list -10
       168         (*fSelectedPostStepDoItVector)[np] = InActivated;
       169         continue;
       170       }   // NULL means the process is inactivated by a user on fly.
       171  
       172       physIntLength = fCurrentProcess->
       173                       PostStepGPIL( *fTrack,
       174                                                   fPreviousStepSize,
       175                                                        &fCondition );




::

    498 inline G4double G4VProcess::PostStepGPIL( const G4Track& track,
    499                                    G4double   previousStepSize,
    500                                    G4ForceCondition* condition )
    501 {
    502   G4double value
    503    =PostStepGetPhysicalInteractionLength(track, previousStepSize, condition);
    504   return thePILfactor*value;
    505 }


::

    simon:cfg4 blyth$ g4-hh PostStepGetPhysicalInteractionLength | grep Scintillation
    simon:cfg4 blyth$ g4-hh PostStepGetPhysicalInteractionLength | grep OpAbsorption
    simon:cfg4 blyth$ 
    simon:cfg4 blyth$ g4-hh PostStepGetPhysicalInteractionLength | wc
          68     322   11105


::

    74 class G4OpAbsorption : public G4VDiscreteProcess

::

     71 G4double G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(
     72                              const G4Track& track,
     73                  G4double   previousStepSize,
     74                  G4ForceCondition* condition
     75                 )
     76 {
     77   if ( (previousStepSize < 0.0) || (theNumberOfInteractionLengthLeft<=0.0)) {
     78     // beggining of tracking (or just after DoIt of this process)
     79     ResetNumberOfInteractionLengthLeft();
     80   } else if ( previousStepSize > 0.0) {
     81     // subtract NumberOfInteractionLengthLeft 
     82     SubtractNumberOfInteractionLengthLeft(previousStepSize);
     83   } else {
     84     // zero step
     85     //  DO NOTHING
     86   }
     87 
     88   // condition is set to "Not Forced"
     ..
     90 
     91   // get mean free path
     92   currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
     93 



Look for this returning negative::

    (lldb) b G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) 


    (lldb) b 109
    Breakpoint 2: where = libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) + 647 at G4VDiscreteProcess.cc:109, address = 0x000000010430af17
    (lldb) c
    Process 36035 resuming
    Process 36035 stopped
    * thread #1: tid = 0x74c2e8, 0x000000010430af17 libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(this=0x000000010933f390, track=0x000000010e9b0d20, previousStepSize=0, condition=0x000000010910e408) + 647 at G4VDiscreteProcess.cc:109, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x000000010430af17 libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(this=0x000000010933f390, track=0x000000010e9b0d20, previousStepSize=0, condition=0x000000010910e408) + 647 at G4VDiscreteProcess.cc:109
       106      G4cout << "InteractionLength= " << value/cm <<"[cm] " <<G4endl;
       107    }
       108  #endif
    -> 109    return value;
       110  }
       111  
       112  G4VParticleChange* G4VDiscreteProcess::PostStepDoIt(
    (lldb) br mod -c 'value < 0'
    (lldb) br lis
    Current breakpoints:
    1: name = 'G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*)', locations = 1, resolved = 1, hit count = 1
      1.1: where = libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) + 35 at G4VDiscreteProcess.cc:77, address = 0x000000010430acb3, resolved, hit count = 1 

    2: file = '/usr/local/env/g4/geant4.10.02/source/processes/management/src/G4VDiscreteProcess.cc', line = 109, locations = 1, resolved = 1, hit count = 1
    Condition: value < 0

      2.1: where = libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*) + 647 at G4VDiscreteProcess.cc:109, address = 0x000000010430af17, resolved, hit count = 1 

    (lldb) br dis 1 
    1 breakpoints disabled.


::

    (lldb) [2016-May-27 20:38:14.840441]:info: CSA tots evt 1 trk 2 stp 2
    [2016-May-27 20:38:14.851408]:info: CSA tots evt 1 trk 2 stp 3
    [2016-May-27 20:38:14.862188]:info: CSA tots evt 1 trk 2 stp 4
    [2016-May-27 20:38:14.874536]:info: CSA tots evt 1 trk 2 stp 5
    [2016-May-27 20:38:14.886434]:info: CSA tots evt 1 trk 2 stp 6
    [2016-May-27 20:38:14.898480]:info: CSA tots evt 1 trk 2 stp 7
    [2016-May-27 20:38:14.910996]:info: CSA tots evt 1 trk 2 stp 8
    Process 36035 stopped
    * thread #1: tid = 0x74c2e8, 0x000000010430af17 libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(this=0x000000010933ef00, track=0x000000010e9b0c10, previousStepSize=0, condition=0x000000010910e408) + 647 at G4VDiscreteProcess.cc:109, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x000000010430af17 libG4processes.dylib`G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(this=0x000000010933ef00, track=0x000000010e9b0c10, previousStepSize=0, condition=0x000000010910e408) + 647 at G4VDiscreteProcess.cc:109
       106      G4cout << "InteractionLength= " << value/cm <<"[cm] " <<G4endl;
       107    }
       108  #endif
    -> 109    return value;
       110  }
       111  
       112  G4VParticleChange* G4VDiscreteProcess::PostStepDoIt(
    (lldb) p value
    (G4double) $31 = -340.79660414597521

    (lldb) p theProcessName
    (G4String) $33 = (std::__1::string = "OpAbsorption")

    (lldb) p currentInteractionLength
    (G4double) $34 = -863.39274890152888
    (lldb) p theNumberOfInteractionLengthLeft
    (G4double) $35 = 0.39471793639634045
    (lldb) p currentInteractionLength*theNumberOfInteractionLengthLeft
    (double) $36 = -340.79660414597521
    (lldb) p value
    (G4double) $37 = -340.79660414597521

    (lldb) b G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) 


    (lldb) b 153
    Breakpoint 2: where = libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(G4Track const&, double, G4ForceCondition*) + 181 at G4OpAbsorption.cc:153, address = 0x000000010430f825
    (lldb) c
    Process 36755 resuming
    Process 36755 stopped
    * thread #1: tid = 0x74cbb0, 0x000000010430f825 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(this=0x000000010e2ef8c0, aTrack=0x000000010f29af80, (null)=0, (null)=0x0000000109030308) + 181 at G4OpAbsorption.cc:153, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x000000010430f825 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(this=0x000000010e2ef8c0, aTrack=0x000000010f29af80, (null)=0, (null)=0x0000000109030308) + 181 at G4OpAbsorption.cc:153
       150  //           G4cout << "No Absorption length specified" << G4endl;
       151          }
       152  
    -> 153          return AttenuationLength;
       154  }
    (lldb) p AttenuationLength
    (G4double) $0 = 18507.865306607295
    (lldb) br mod -c 'AttenuationLength < 0'

    ## how to break at the exit of a method ?

::


    (lldb) br dis 1
    1 breakpoints disabled.
    (lldb) c
    Process 36755 resuming
    [2016-May-27 20:44:52.802492]:info: CSA tots evt 1 trk 2 stp 2
    (lldb) [2016-May-27 20:44:52.897328]:info: CSA tots evt 1 trk 2 stp 3
    [2016-May-27 20:44:52.900530]:info: CSA tots evt 1 trk 2 stp 4
    [2016-May-27 20:44:52.903555]:info: CSA tots evt 1 trk 2 stp 5
    [2016-May-27 20:44:52.906583]:info: CSA tots evt 1 trk 2 stp 6
    [2016-May-27 20:44:52.909446]:info: CSA tots evt 1 trk 2 stp 7
    [2016-May-27 20:44:52.912243]:info: CSA tots evt 1 trk 2 stp 8
    Process 36755 stopped
    * thread #1: tid = 0x74cbb0, 0x000000010430f825 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(this=0x000000010e2ef8c0, aTrack=0x000000010f29ae70, (null)=0, (null)=0x0000000109030308) + 181 at G4OpAbsorption.cc:153, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x000000010430f825 libG4processes.dylib`G4OpAbsorption::GetMeanFreePath(this=0x000000010e2ef8c0, aTrack=0x000000010f29ae70, (null)=0, (null)=0x0000000109030308) + 181 at G4OpAbsorption.cc:153
       150  //           G4cout << "No Absorption length specified" << G4endl;
       151          }
       152  
    -> 153          return AttenuationLength;
       154  }
    (lldb) p AttenuationLength
    (G4double) $8 = -863.39274890152888
    (lldb) p aMaterial
    (const G4Material *) $9 = 0x000000010a615870
    (lldb) p *aMaterial
    (const G4Material) $10 = {
      fName = (std::__1::string = "/dd/Materials/GdDopedLS")
      fChemicalFormula = (std::__1::string = "")
      fDensity = 5.368943773533483E+18
      fState = kStateSolid



    lldb) p aMaterial->fMaterialPropertiesTable
    (G4MaterialPropertiesTable *const) $11 = 0x000000010e28d480
    (lldb) p *aMaterial->fMaterialPropertiesTable
    (G4MaterialPropertiesTable) $12 = {
      MPT = size=7 {
        [0] = {
          __cc = {
            first = (std::__1::string = "ABSLENGTH")
            second = 0x000000010e28e770
          }
          __nc = {
            first = (std::__1::string = "ABSLENGTH")
            second = 0x000000010e28e770
          }
        }
        [1] = {
          __cc = {
            first = (std::__1::string = "FASTCOMPONENT")
            second = 0x000000010e28faf0
          }
          __nc = {
            first = (std::__1::string = "FASTCOMPONENT")
            second = 0x000000010e28faf0
          }



How did this manage to arrive at a negative value ?::

    (lldb) p AttenuationLengthVector
    (G4MaterialPropertyVector *) $13 = 0x000000010e28e770
    (lldb) p *AttenuationLengthVector
    (G4MaterialPropertyVector) $14 = {
      G4PhysicsVector = {
        type = T_G4PhysicsOrderedFreeVector
        edgeMin = 0.0000015120022870975581
        edgeMax = 0.000020664031256999959
        numberOfNodes = 39
        dataVector = size=39 {
          [0] = 2021.013916015625
          [1] = 3358.37451171875
          [2] = 3910.525390625
          [3] = 989.15386962890625
          [4] = 1876.99755859375
          [5] = 2573.489990234375
          [6] = 4617.7197265625
          [7] = 6944.9453125
          [8] = 7315.3310546875
          [9] = 5387.97998046875
          [10] = 14952.751953125
          [11] = 14692.2119140625
          [12] = 21527.98828125
          [13] = 27079.125
          [14] = 27572.681640625
          [15] = 26137.435546875
          [16] = 33867.41796875
          [17] = 27410.125
          [18] = 26623.083984375
          [19] = 28043.51953125
          [20] = 12864.24609375
          [21] = 72.760719299316406
          [22] = 4.1327381134033203
          [23] = 1.9125092029571533
          [24] = 0.46045231819152832
          [25] = 0.394828200340271
          [26] = 0.32920405268669128
          [27] = 0.26357993483543396
          [28] = 0.19795580208301544
          [29] = 0.13233168423175812
          [30] = 0.066707544028759003
          [31] = 0.0010834120912477374
          [32] = 0.0010000000474974513
          [33] = 0.0010000000474974513
          [34] = 0.0010000000474974513
          [35] = 0.0010000000474974513
          [36] = 0.0010000000474974513
          [37] = 0.0010000000474974513
          [38] = 0.0010000000474974513
        }
        binVector = size=39 {
          [0] = 0.0000015120022870975581
          [1] = 0.0000015498023442749969
          [2] = 0.0000015895408659230739
          [3] = 0.000001631370888710523
          [4] = 0.0000016754619938108077
          [5] = 0.0000017220026047499965
          [6] = 0.000001771202679171425
          [7] = 0.0000018232968756176433
          [8] = 0.0000018785482960909054
          [9] = 0.0000019372529303437461
          [10] = 0.0000019997449603548349
          [11] = 0.0000020664031256999961
          [12] = 0.0000021376584058965476
          [13] = 0.0000022140033489642815
          [14] = 0.0000022960034729999953
          [15] = 0.0000023843112988846106
          [16] = 0.0000024796837508399948
          [17] = 0.0000025830039071249949
          [18] = 0.0000026953084248260815
          [19] = 0.0000028178224441363579
          [20] = 0.0000029520044652857083
          [21] = 0.0000030996046885499937
          [22] = 0.000003262741777421046
          [23] = 0.0000034440052094999931
          [24] = 0.0000036465937512352865
          [25] = 0.0000038745058606874922
          [26] = 0.0000041328062513999922
          [27] = 0.0000044280066979285631
          [28] = 0.0000047686225977692212
          [29] = 0.0000051660078142499898
          [30] = 0.0000056356448882727159
          [31] = 0.0000061992093770999875
          [32] = 0.0000068880104189999862
          [33] = 0.0000077490117213749843
          [34] = 0.0000088560133958571262
          [35] = 0.00001033201562849998
          [36] = 0.000012398418754199975
          [37] = 0.000015498023442749969
          [38] = 0.000020664031256999959
        }
        secDerivative = size=39 {
          [0] = 4.0542779613141545E+18
          [1] = 41560180835040992
          [2] = -4.1769379986430454E+18
          [3] = 4.6202661727990088E+18
          [4] = -1.6230783113708938E+18
          [5] = 1.2610884519878971E+18
          [6] = -1.3036740408857141E+17
          [7] = -3.2424055012097984E+17
          [8] = -2.7404852975083121E+18
          [9] = 6.9039567958456658E+18
          [10] = -6.3571485696712888E+18
          [11] = 4.0329326730502687E+18
          [12] = -1.2572508734853816E+18
          [13] = -7.2772168789968858E+17
          [14] = -8.9887753853942016E+17
          [15] = 2.6295937816179963E+18
          [16] = -3.175046056580545E+18
          [17] = 1.4482964449061117E+18
          [18] = 3.2414150115403514E+17
          [19] = -1.6590940890374528E+18
          [20] = 4.7504197691129018E+17
          [21] = 7.7009767106113114E+17
          [22] = -1.9158642120259072E+17
          [23] = 48484578267391344
          [24] = -12168231653967552
          [25] = 3054687160580781
          [26] = -763158528100454.75
          [27] = 189824270585361.22
          [28] = -46806807523797.555
          [29] = 11579759518210.941
          [30] = -2830809757356.4458
          [31] = 979368172282.72241
          [32] = -232428772359.01233
          [33] = 54315438042.016289
          [34] = -12414060800.327751
          [35] = 2735250775.957417
          [36] = -518180375.24327075
          [37] = -93802467.696051672
          [38] = 613494044.88264692
        }
        useSpline = true
        dBin = 0
        baseBin = 0
        verboseLevel = 0
      }
    }
    (lldb) p thePhotonMomentum
    (G4double) $15 = 0.0000031401392360999671



::

    In [1]: a = np.load("ABSLENGTH.npy")

    In [2]: a
    Out[2]: 
    array([[    60.   ,      0.001],
           [    80.   ,      0.001],
           [   100.   ,      0.001],
           [   120.   ,      0.001],
           [   140.   ,      0.001],
           [   160.   ,      0.001],
           [   180.   ,      0.001],
           [   200.   ,      0.001],
           [   220.   ,      0.067],
           [   240.   ,      0.132],
           [   260.   ,      0.198],
           [   280.   ,      0.264],
           [   300.   ,      0.329],
           [   320.   ,      0.395],
           [   340.   ,      0.46 ],
           [   360.   ,      1.913],
           [   380.   ,      4.133],
           [   400.   ,     72.761],
           [   420.   ,  12864.246],
           [   440.   ,  28043.52 ],
           [   460.   ,  26623.084],
           [   480.   ,  27410.125],
           [   500.   ,  33867.418],
           [   520.   ,  26137.436],
           [   540.   ,  27572.682],
           [   560.   ,  27079.125],
           [   580.   ,  21527.988],
           [   600.   ,  14692.212],
           [   620.   ,  14952.752],
           [   640.   ,   5387.98 ],
           [   660.   ,   7315.331],
           [   680.   ,   6944.945],
           [   700.   ,   4617.72 ],
           [   720.   ,   2573.49 ],
           [   740.   ,   1876.998],
           [   760.   ,    989.154],
           [   780.   ,   3910.525],
           [   800.   ,   3358.375],
           [   820.   ,   2021.014]], dtype=float32)


    In [3]: pwd
    Out[3]: u'/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GdDopedLS'


::

    (lldb) expr -- for(int i=60 ; i < 840 ; i+= 20) (void)printf("%d %10.3f \n", i, AttenuationLengthVector->Value(1239.84193*1e-6/float(i)))
    60      0.001 
    80      0.001 
    100      0.001 
    120      0.001 
    140      0.001 
    160      0.001 
    180      0.001 
    200      0.001 
    220      0.067 
    240      0.132 
    260      0.198 
    280      0.264 
    300      0.329 
    320      0.395 
    340      0.461 
    360      1.912 
    380      4.134 
    400     72.756 
    420  12864.229 
    440  28043.513 
    460  26623.088 
    480  27410.117 
    500  33867.420 
    520  26137.441 
    540  27572.679 
    560  27079.129 
    580  21527.999 
    600  14692.213 
    620  14952.760 
    640   5387.986 
    660   7315.327 
    680   6944.948 
    700   4617.724 
    720   2573.492 
    740   1877.000 
    760    989.151 
    780   3910.522 
    800   3358.377 
    820   2021.013 



Smth wrong with the interpolation::

    (lldb) expr -- for(int i=300 ; i < 410 ; i+= 5) (void)printf("%d %10.3f \n", i, AttenuationLengthVector->Value(1239.84193*1e-6/float(i)))
    300      0.329 
    305     -5.097 
    310     -9.354 
    315     -8.572 
    320      0.395 
    325     17.200 
    330     30.487 
    335     28.215 
    340      0.461 
    345    -51.879 
    350    -93.353 
    355    -85.979 
    360      1.912 
    365    168.124 
    370    301.139 
    375    280.349 
    380      4.134 
    385   -521.067 
    390   -935.974 
    395   -849.023 
    400     72.756 
    405   2037.167 





::

    hc 1239.84193 eV nm  


Dont use spline for ABSLENGTH::

    (lldb) p AttenuationLengthVector->useSpline
    (G4bool) $60 = true
    (lldb) expr --  AttenuationLengthVector->useSpline = false
    (G4bool) $61 = false
    (lldb) p AttenuationLengthVector->useSpline
    (G4bool) $62 = false
    (lldb) expr -- for(int i=300 ; i < 410 ; i+= 5) (void)printf("%d %10.3f \n", i, AttenuationLengthVector->Value(1239.84193*1e-6/float(i)))
    300      0.329 
    305      0.346 
    310      0.363 
    315      0.379 
    320      0.395 
    325      0.412 
    330      0.429 
    335      0.445 
    340      0.460 
    345      0.839 
    350      1.207 
    355      1.565 
    360      1.913 
    365      2.490 
    370      3.053 
    375      3.600 
    380      4.133 
    385     21.958 
    390     39.327 
    395     56.255 
    400     72.761 
    405   3389.060 
    (lldb) expr -- for(int i=60 ; i < 840 ; i+= 20) (void)printf("%d %10.3f \n", i, AttenuationLengthVector->Value(1239.84193*1e-6/float(i)))
    60      0.001 
    80      0.001 
    100      0.001 
    120      0.001 
    140      0.001 
    160      0.001 
    180      0.001 
    200      0.001 
    220      0.067 
    240      0.132 
    260      0.198 
    280      0.264 
    300      0.329 
    320      0.395 
    340      0.460 
    360      1.913 
    380      4.133 
    400     72.761 
    420  12864.235 
    440  28043.505 
    460  26623.085 
    480  27410.124 
    500  33867.411 
    520  26137.444 
    540  27572.680 
    560  27079.126 
    580  21527.995 
    600  14692.221 
    620  14952.752 
    640   5387.993 
    660   7315.328 
    680   6944.946 
    700   4617.723 
    720   2573.493 
    740   1876.999 
    760    989.155 
    780   3910.521 
    800   3358.375 
    820   2021.016 
    (lldb) 



