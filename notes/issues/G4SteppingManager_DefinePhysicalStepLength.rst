G4SteppingManager_DefinePhysicalStepLength
==============================================

* from :doc:`ideas_on_random_alignment_in_new_workflow`


::

    116 G4StepStatus G4SteppingManager::Stepping()
    ...
    136    fStep->CopyPostToPreStepPoint();
    137    fStep->ResetTotalEnergyDeposit();
    ...
    179      // Find minimum Step length demanded by active disc./cont. processes
    180      DefinePhysicalStepLength();
    181 
    182      // Store the Step length (geometrical length) to G4Step and G4Track
    183      fStep->SetStepLength( PhysicalStep );
    184      fTrack->SetStepLength( PhysicalStep );
    185      G4double GeomStepLength = PhysicalStep;

    /// PhysicalStep is the winning smallest step length from the competing processes including G4Transportation 

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



::

    128  void G4SteppingManager::DefinePhysicalStepLength()
    ...
    133    PhysicalStep  = DBL_MAX;          // Initialize by a huge number    
    134    physIntLength = DBL_MAX;          // Initialize by a huge number   

    ...
    162 // GPIL for PostStep
    163    fPostStepDoItProcTriggered = MAXofPostStepLoops;
    164 
    165    for(size_t np=0; np < MAXofPostStepLoops; np++){
    166      fCurrentProcess = (*fPostStepGetPhysIntVector)(np);
    167      if (fCurrentProcess== 0) {
    168        (*fSelectedPostStepDoItVector)[np] = InActivated;
    169        continue;
    170      }   // NULL means the process is inactivated by a user on fly.
    171 
    172      physIntLength = fCurrentProcess->
    173                      PostStepGPIL( *fTrack,
    174                                                  fPreviousStepSize,
    175                                                       &fCondition );



::

    (lldb) c
    Process 43774 resuming
    2022-06-20 10:34:04.864 INFO  [27180517] [U4Random::flat@416]  m_seq_index    9 m_seq_nv  256 cursor    0 idx 2304 d    0.51319
    Process 43774 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 3.1
        frame #0: 0x0000000101f1ba4a libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x0000000106fbe7b0) at G4SteppingManager2.cc:181
       178 	     if(verboseLevel>0) fVerbose->DPSLPostStep();
       179 	#endif
       180 	
    -> 181 	     switch (fCondition) {
       182 		 case ExclusivelyForced:
       183 		     (*fSelectedPostStepDoItVector)[np] = ExclusivelyForced;
       184 		     fStepStatus = fExclusivelyForcedProc;
    Target 0: (U4RecorderTest) stopped.
    (lldb) p physIntLength 
    (G4double) $8 = 1.7976931348623157E+308
    (lldb) p fCurrentProcess
    (DsG4Scintillation *) $9 = 0x0000000108a38ab0
    (lldb) p fCondition
    (G4ForceCondition) $10 = StronglyForced
    (lldb) 


Returning DBL_MAX means the process will never define the step so the 
corresponding random throw is just a burn, doing nothing.
But ordering+count of the burns are still relevant as need to match them 
with GPU side burns.


Only G4OpAbsorption and G4OpRayleigh give actual lengths::


    1078 G4double DsG4Scintillation::GetMeanFreePath(const G4Track&,
    1079                                             G4double ,
    1080                                             G4ForceCondition* condition)
    1081 {
    1082     *condition = StronglyForced;
    1083 
    1084     return DBL_MAX;
    1085 
    1086 }

    1419 G4double InstrumentedG4OpBoundaryProcess::GetMeanFreePath(const G4Track& ,
    1420                                               G4double ,
    1421                                               G4ForceCondition* condition)
    1422 {
    1423   *condition = Forced;
    1424 
    1425   return DBL_MAX;
    1426 }


    122 G4double G4OpAbsorption::GetMeanFreePath(const G4Track& aTrack,
    123                          G4double ,
    124                          G4ForceCondition* )
    125 {
    ...
    139        AttenuationLengthVector = aMaterialPropertyTable->
    140                                                 GetProperty(kABSLENGTH);
    141            if ( AttenuationLengthVector ){
    142              AttenuationLength = AttenuationLengthVector->
    143                                          Value(thePhotonMomentum);
    ...
    153         return AttenuationLength;
    154 }

    248 G4double G4OpRayleigh::GetMeanFreePath(const G4Track& aTrack,
    249                                        G4double ,
    250                                        G4ForceCondition* )
    251 {
    ...
    256   G4PhysicsOrderedFreeVector* rayleigh =
    257                               static_cast<G4PhysicsOrderedFreeVector*>
    258                               ((*thePhysicsTable)(material->GetIndex()));
    259 
    260   G4double rsLength = DBL_MAX;
    261   if( rayleigh != NULL ) rsLength = rayleigh->Value( photonMomentum );
    262   return rsLength;
    263 }


::

    (lldb) p fCondition
    (G4ForceCondition) $13 = NotForced
    (lldb) p physIntLength
    (G4double) $14 = 10649.056684275361
    (lldb) p fCurrentProcess
    (G4OpRayleigh *) $15 = 0x0000000108a3b530
    (lldb) 

    (lldb) p PhysicalStep       ## starts as DBL_MAX : so will get set to first 
    (G4double) $16 = 1.7976931348623157E+308
    (lldb) 

    (lldb) p fCondition
    (G4ForceCondition) $17 = NotForced
    (lldb) p physIntLength
    (G4double) $18 = 3206.1502470764271
    (lldb) p fCurrentProcess
    (G4OpAbsorption *) $19 = 0x0000000108a3b3b0
    (lldb) 
    (lldb) p PhysicalStep
    (G4double) $20 = 10649.056684275361
    (lldb) 


    (lldb) p physIntLength
    (G4double) $21 = 1.7976931348623157E+308
    (lldb) p fCurrentProcess
    (G4Transportation *) $22 = 0x0000000108a055e0
    (lldb) p fCondition
    (G4ForceCondition) $23 = Forced
    (lldb) 


    (lldb) c
    Process 43774 resuming
    Process 43774 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 6.1
        frame #0: 0x0000000101f1bcb2 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x0000000106fbe7b0) at G4SteppingManager2.cc:225
       222 	
       223 	   }
       224 	
    -> 225 	   if (fPostStepDoItProcTriggered<MAXofPostStepLoops) {
       226 	       if ((*fSelectedPostStepDoItVector)[fPostStepDoItProcTriggered] == 
       227 		   InActivated) {
       228 		   (*fSelectedPostStepDoItVector)[fPostStepDoItProcTriggered] = 
    Target 0: (U4RecorderTest) stopped.
    (lldb) p PhysicalStep
    (G4double) $24 = 3206.1502470764271
    (lldb) 


Note the geometry distance does not enter in the first loop over processes, G4Transportation giving DBL_MAX.
The geometry distance enters in the second loop over processes. 

g4-cls G4Transportation::

    087      G4double PostStepGetPhysicalInteractionLength(
     88                              const G4Track& ,
     89                              G4double   previousStepSize,
     90                              G4ForceCondition* pForceCond
     91                             );
     92        // Forces the PostStepDoIt action to be called, 
     93        // but does not limit the step.

    638 //  This ensures that the PostStep action is always called,
    639 //  so that it can do the relocation if it is needed.
    640 // 
    641 
    642 G4double G4Transportation::
    643 PostStepGetPhysicalInteractionLength( const G4Track&,
    644                                             G4double, // previousStepSize
    645                                             G4ForceCondition* pForceCond )
    646 {
    647   fFieldExertedForce = false; // Not known
    648   *pForceCond = Forced ;
    649   return DBL_MAX ;  // was kInfinity ; but convention now is DBL_MAX
    650 }
    651 


g4-cls G4SteppingManager ctd::

    181      switch (fCondition) {
    182      case ExclusivelyForced:
    183          (*fSelectedPostStepDoItVector)[np] = ExclusivelyForced;
    184          fStepStatus = fExclusivelyForcedProc;
    185          fStep->GetPostStepPoint()
    186          ->SetProcessDefinedStep(fCurrentProcess);
    187          break;
    188      case Conditionally:
    189        //        (*fSelectedPostStepDoItVector)[np] = Conditionally;
    190          G4Exception("G4SteppingManager::DefinePhysicalStepLength()", "Tracking1001", FatalException, "This feature no more supported");
    191 
    192          break;
    193      case Forced:
    194          (*fSelectedPostStepDoItVector)[np] = Forced;
    195          break;
    196      case StronglyForced:
    197          (*fSelectedPostStepDoItVector)[np] = StronglyForced;
    198          break;
    199      default:
    200          (*fSelectedPostStepDoItVector)[np] = InActivated;
    201          break;
    202      }
    203 
    204 
    205 
    206      if (fCondition==ExclusivelyForced) {
    207      for(size_t nrest=np+1; nrest < MAXofPostStepLoops; nrest++){
    208          (*fSelectedPostStepDoItVector)[nrest] = InActivated;
    209      }
    210      return;  // Take note the 'return' at here !!! 
    211      }
    212      else{
    213      if(physIntLength < PhysicalStep ){
    214          PhysicalStep = physIntLength;
    215          fStepStatus = fPostStepDoItProc;
    216          fPostStepDoItProcTriggered = G4int(np);
    217          fStep->GetPostStepPoint()
    218          ->SetProcessDefinedStep(fCurrentProcess);
    219      }
    220      }
    221     

* smallest physIntLength wins, becoming PhysicalStep : this is picking between scatter/absorption/sail-to-boundary



::

    (lldb) p (*fSelectedPostStepDoItVector)
    (G4SelectedPostStepDoItVector) $28 = size=100 {
      [0] = 5
      [1] = 1
      [2] = 0
      [3] = 0
      [4] = 1
      [5] = 0
      [6] = 0


::

    233 // GPIL for AlongStep
    234    proposedSafety = DBL_MAX;
    235    G4double safetyProposedToAndByProcess = proposedSafety;
    236 
    237    for(size_t kp=0; kp < MAXofAlongStepLoops; kp++){
    238      fCurrentProcess = (*fAlongStepGetPhysIntVector)[kp];

    (lldb) p MAXofAlongStepLoops
    (size_t) $30 = 1
    (lldb) 

       237 	   for(size_t kp=0; kp < MAXofAlongStepLoops; kp++){
       238 	     fCurrentProcess = (*fAlongStepGetPhysIntVector)[kp];
    -> 239 	     if (fCurrentProcess== 0) continue;
       240 	         // NULL means the process is inactivated by a user on fly.
       241 	
       242 	     physIntLength = fCurrentProcess->
    Target 0: (U4RecorderTest) stopped.
    (lldb) p fCurrentProcess
    (G4Transportation *) $32 = 0x0000000108a055e0
    (lldb) 




    239      if (fCurrentProcess== 0) continue;
    240          // NULL means the process is inactivated by a user on fly.
    241 
    242      physIntLength = fCurrentProcess->
    243                      AlongStepGPIL( *fTrack, fPreviousStepSize,
    244                                      PhysicalStep,
    245                      safetyProposedToAndByProcess,
    246                                     &fGPILSelection );



::

    134       virtual G4double AlongStepGetPhysicalInteractionLength(
    135                              const G4Track& track,
    136                  G4double  previousStepSize,
    137                  G4double  currentMinimumStep,
    138                  G4double& proposedSafety,
    139                              G4GPILSelection* selection) = 0;
    140 


    482 inline G4double G4VProcess::AlongStepGPIL( const G4Track& track,
    483                                      G4double  previousStepSize,
    484                                      G4double  currentMinimumStep,
    485                                      G4double& proposedSafety,
    486                                      G4GPILSelection* selection     )
    487 {
    488   G4double value
    489    =AlongStepGetPhysicalInteractionLength(track, previousStepSize, currentMinimumStep, proposedSafety, selection);
    490   return value;
    491 }


    149      G4Navigator*         fLinearNavigator;

    118   fLinearNavigator = transportMgr->GetNavigatorForTracking() ;


    156 //////////////////////////////////////////////////////////////////////////
    157 //
    158 // Responsibilities:
    159 //    Find whether the geometry limits the Step, and to what length
    160 //    Calculate the new value of the safety and return it.
    161 //    Store the final time, position and momentum.
    162 
    163 G4double G4Transportation::
    164 AlongStepGetPhysicalInteractionLength( const G4Track&  track,
    165                                              G4double, //  previousStepSize
    166                                              G4double  currentMinimumStep,
    167                                              G4double& currentSafety,
    168                                              G4GPILSelection* selection )
    169 {
    170   G4double geometryStepLength= -1.0, newSafety= -1.0;
    171   fParticleIsLooping = false ;
    172 
    ...
    273      {
    274        //  Find whether the straight path intersects a volume
    275        //
    276        linearStepLength = fLinearNavigator->ComputeStep( startPosition,
    277                                                          startMomentumDir,
    278                                                          currentMinimumStep,
    279                                                          newSafety) ;
    280        // Remember last safety origin & value.
    281        //
    282        fPreviousSftOrigin = startPosition ;
    283        fPreviousSafety    = newSafety ;
    284        fpSafetyHelper->SetCurrentSafety( newSafety, startPosition);
    285 
    286        currentSafety = newSafety ;
    287 
    288        fGeometryLimitedStep= (linearStepLength <= currentMinimumStep);
    289        if( fGeometryLimitedStep )
    290        {
    291          // The geometry limits the Step size (an intersection was found.)
    292          geometryStepLength   = linearStepLength ;
    293        }
    294        else
    295        {
    296          // The full Step is taken.
    297          geometryStepLength   = currentMinimumStep ;
    298        }
    299      }
    300      fEndPointDistance = geometryStepLength ;
    301 
    302      // Calculate final position
    303      //
    304      fTransportEndPosition = startPosition+geometryStepLength*startMomentumDir ;
    305 
    306      // Momentum direction, energy and polarisation are unchanged by transport
    307      //
    308      fTransportEndMomentumDir   = startMomentumDir ;
    309      fTransportEndKineticEnergy = track.GetKineticEnergy() ;
    310      fTransportEndSpin          = track.GetPolarization();
    311      fParticleIsLooping         = false ;
    312      fMomentumChanged           = false ;
    313      fEndGlobalTimeComputed     = false ;






Here comes the geometry length of 49mm from G4Transportation::AlongStepGPIL::

    (lldb) c
    Process 43774 resuming
    Process 43774 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 9.1
        frame #0: 0x0000000101f1be26 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x0000000106fbe7b0) at G4SteppingManager2.cc:251
       248 	                         // !!!!! Verbose
       249 	     if(verboseLevel>0) fVerbose->DPSLAlongStep();
       250 	#endif
    -> 251 	     if(physIntLength < PhysicalStep){
       252 	       PhysicalStep = physIntLength;
       253 	
       254 	       // Check if the process wants to be the GPIL winner. For example,
    Target 0: (U4RecorderTest) stopped.
    (lldb) p physIntLength
    (G4double) $33 = 49
    (lldb) 




    247 #ifdef G4VERBOSE
    248                          // !!!!! Verbose
    249      if(verboseLevel>0) fVerbose->DPSLAlongStep();
    250 #endif
    251      if(physIntLength < PhysicalStep){
    252        PhysicalStep = physIntLength;
    253 
    254        // Check if the process wants to be the GPIL winner. For example,
    255        // multi-scattering proposes Step limit, but won't be the winner.
    256        if(fGPILSelection==CandidateForSelection){
    257           fStepStatus = fAlongStepDoItProc;
    258           fStep->GetPostStepPoint()
    259                ->SetProcessDefinedStep(fCurrentProcess);
    260        }
    261 
    262           // Transportation is assumed to be the last process in the vector
    263        if(kp == MAXofAlongStepLoops-1)
    264       fStepStatus = fGeomBoundary;
    265      }
    266 
    267      // Make sure to check the safety, even if Step is not limited 
    268      //  by this process.                      J. Apostolakis, June 20, 1998
    269      // 
    270      if (safetyProposedToAndByProcess < proposedSafety)
    271         // proposedSafety keeps the smallest value:
    272         proposedSafety               = safetyProposedToAndByProcess;
    273      else
    274         // safetyProposedToAndByProcess always proposes a valid safety:
    275         safetyProposedToAndByProcess = proposedSafety;
    276      
    277    }
    278 } // void G4SteppingManager::DefinePhysicalStepLength() //
    279 





g4-cls G4VProcess::


    146       virtual G4double PostStepGetPhysicalInteractionLength(
    147                              const G4Track& track,
    148                  G4double   previousStepSize,
    149                  G4ForceCondition* condition
    150                 ) = 0;
    ...
    186 
    187       // These three GPIL methods are used by Stepping Manager.
    188       // They invoke virtual GPIL methods listed above.
    189       // As for AtRest and PostStep the returned value is multipled by thePILfactor 
    190       // 
    191       G4double AlongStepGPIL( const G4Track& track,
    192                               G4double  previousStepSize,
    193                               G4double  currentMinimumStep,
    194                               G4double& proposedSafety,
    195                               G4GPILSelection* selection     );
    196 
    197       G4double AtRestGPIL( const G4Track& track,
    198                            G4ForceCondition* condition );
    199 
    200       G4double PostStepGPIL( const G4Track& track,
    201                              G4double   previousStepSize,
    202                              G4ForceCondition* condition );
    203 
    ...
    501 inline G4double G4VProcess::PostStepGPIL( const G4Track& track,
    502                                    G4double   previousStepSize,
    503                                    G4ForceCondition* condition )
    504 {
    505   G4double value
    506    =PostStepGetPhysicalInteractionLength(track, previousStepSize, condition);
    507   return thePILfactor*value;
    508 }




g4-cls G4VDiscreteProcess::

     58 class G4VDiscreteProcess : public G4VProcess

     54 G4VDiscreteProcess::G4VDiscreteProcess(const G4String& aName , G4ProcessType aType)
     55                   : G4VProcess(aName, aType)
     56 {
     57   enableAtRestDoIt = false;
     58   enableAlongStepDoIt = false;
     59 
     60 }

::

    071 G4double G4VDiscreteProcess::PostStepGetPhysicalInteractionLength(
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
     89   *condition = NotForced;
     90 
     91   // get mean free path
     92   currentInteractionLength = GetMeanFreePath(track, previousStepSize, condition);
     93 
     94   G4double value;
     95   if (currentInteractionLength <DBL_MAX) {
     96     value = theNumberOfInteractionLengthLeft * currentInteractionLength;
     97   } else {
     98     value = DBL_MAX;
     99   }
    ... 
    109   return value;
    110 }


    096 void G4VProcess::ResetNumberOfInteractionLengthLeft()
     97 {
     98   theNumberOfInteractionLengthLeft =  -1.*G4Log( G4UniformRand() );
     99   theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft;
    100 }

    546 inline
    547 void G4VProcess::SubtractNumberOfInteractionLengthLeft(
    548                                   G4double previousStepSize )
    549 {
    550   if (currentInteractionLength>0.0) {
    551     theNumberOfInteractionLengthLeft -= previousStepSize/currentInteractionLength;
    552     if(theNumberOfInteractionLengthLeft<0.) {
    553        theNumberOfInteractionLengthLeft=CLHEP::perMillion;
    554     }
    555 
    556   } else {
    557 #ifdef G4VERBOSE
    558     if (verboseLevel>0) {
    559       G4cerr << "G4VProcess::SubtractNumberOfInteractionLengthLeft()";
    560       G4cerr << " [" << theProcessName << "]" <<G4endl;
    561       G4cerr << " currentInteractionLength = " << currentInteractionLength << " [mm]";
    562       G4cerr << " previousStepSize = " << previousStepSize << " [mm]";
    563       G4cerr << G4endl;
    564     }
    565 #endif
    566     G4String msg = "Negative currentInteractionLength for ";
    567     msg +=      theProcessName;
    568     G4Exception("G4VProcess::SubtractNumberOfInteractionLengthLeft()",
    569                 "ProcMan201",EventMustBeAborted,
    570                 msg);
    571   }
    572 }



    102 void G4VProcess::StartTracking(G4Track*)
    103 {
    104   currentInteractionLength = -1.0;
    105   theNumberOfInteractionLengthLeft = -1.0;
    106   theInitialNumberOfInteractionLength=-1.0;
    112 }
    113 
    114 void G4VProcess::EndTracking()
    115 {
    121   theNumberOfInteractionLengthLeft = -1.0;
    122   currentInteractionLength = -1.0;
    123   theInitialNumberOfInteractionLength=-1.0;
    124 }







::

    u4t
    BP=G4SteppingManager::DefinePhysicalStepLength ./U4RecorderTest.sh dbg 


    (lldb) c
    Process 43774 resuming
    Process 43774 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x0000000101f1b95b libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x0000000106fbe7b0) at G4SteppingManager2.cc:165
       162 	// GPIL for PostStep
       163 	   fPostStepDoItProcTriggered = MAXofPostStepLoops;
       164 	
    -> 165 	   for(size_t np=0; np < MAXofPostStepLoops; np++){
       166 	     fCurrentProcess = (*fPostStepGetPhysIntVector)(np);
       167 	     if (fCurrentProcess== 0) {
       168 	       (*fSelectedPostStepDoItVector)[np] = InActivated;
    Target 0: (U4RecorderTest) stopped.
    (lldb) p MAXofPostStepLoops
    (size_t) $0 = 5
    (lldb) 


    (lldb) p *fPostStepGetPhysIntVector
    (G4ProcessVector) $2 = {
      pProcVector = 0x0000000108a02d90 size=5
    }
    (lldb) p (*fPostStepGetPhysIntVector)(0)
    (DsG4Scintillation *) $3 = 0x0000000108a38ab0
    (lldb) p (*fPostStepGetPhysIntVector)(1)
    (InstrumentedG4OpBoundaryProcess *) $4 = 0x0000000108a3b6c0
    (lldb) p (*fPostStepGetPhysIntVector)(2)
    (G4OpRayleigh *) $5 = 0x0000000108a3b530
    (lldb) p (*fPostStepGetPhysIntVector)(3)
    (G4OpAbsorption *) $6 = 0x0000000108a3b3b0
    (lldb) p (*fPostStepGetPhysIntVector)(4)
    (G4Transportation *) $7 = 0x0000000108a055e0
    (lldb) 



What controls the order of these processes ? From u4/tests/U4Physics.cc I expected boundary to be just before Transport::

    206 
    207         if ( fScintillation && fScintillation->IsApplicable(*particle))
    208         {
    209             pmanager->AddProcess(fScintillation);
    210             pmanager->SetProcessOrderingToLast(fScintillation, idxAtRest);
    211             pmanager->SetProcessOrderingToLast(fScintillation, idxPostStep);
    212         }
    213 
    214         if (particleName == "opticalphoton")
    215         {
    216             pmanager->AddDiscreteProcess(fAbsorption);
    217             pmanager->AddDiscreteProcess(fRayleigh);
    218             //pmanager->AddDiscreteProcess(fMieHGScatteringProcess);
    219             pmanager->AddDiscreteProcess(fBoundary);
    220         }
    221     }

