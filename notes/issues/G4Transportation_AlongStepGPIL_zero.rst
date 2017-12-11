G4Transportation_AlongStepGPIL_zero
====================================


Where to burn ?
-----------------

*  chicken-egg ?






G4Transportation::AlongStepGPIL returning zero ?
---------------------------------------------------

::

   g4-;g4-cls G4Transportation



G4VProcess::AlongStepGPIL
---------------------------

::

   g4-;g4-cls G4VProcess

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

    479 inline G4double G4VProcess::AlongStepGPIL( const G4Track& track,
    480                                      G4double  previousStepSize,
    481                                      G4double  currentMinimumStep,
    482                                      G4double& proposedSafety,
    483                                      G4GPILSelection* selection     )
    484 {
    485   G4double value
    486    =AlongStepGetPhysicalInteractionLength(track, previousStepSize, currentMinimumStep, proposedSafety, selection);
    487   return value;
    488 }




::

    // g4-;g4-cls G4SteppingManager

    233 // GPIL for AlongStep
    234    proposedSafety = DBL_MAX;
    235    G4double safetyProposedToAndByProcess = proposedSafety;
    236 
    237    for(size_t kp=0; kp < MAXofAlongStepLoops; kp++){
    238      fCurrentProcess = (*fAlongStepGetPhysIntVector)[kp];
    239      if (fCurrentProcess== 0) continue;
    240          // NULL means the process is inactivated by a user on fly.
    241 
    242      physIntLength = fCurrentProcess->
    243                      AlongStepGPIL( *fTrack, fPreviousStepSize,
    244                                      PhysicalStep,
    245                      safetyProposedToAndByProcess,
    246                                     &fGPILSelection );
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

    // g4-;g4-cls G4Navigator


     73 class G4Navigator
     74 {
    ...
     85   virtual G4double ComputeStep(const G4ThreeVector &pGlobalPoint,
     86                                const G4ThreeVector &pDirection,
     87                                const G4double pCurrentProposedStepLength,
     88                                      G4double  &pNewSafety);
     89     // Calculate the distance to the next boundary intersected
     90     // along the specified NORMALISED vector direction and
     91     // from the specified point in the global coordinate
     92     // system. LocateGlobalPointAndSetup or LocateGlobalPointWithinVolume 
     93     // must have been called with the same global point prior to this call.
     94     // The isotropic distance to the nearest boundary is also
     95     // calculated (usually an underestimate). The current
     96     // proposed Step length is used to avoid intersection
     97     // calculations: if it can be determined that the nearest
     98     // boundary is >pCurrentProposedStepLength away, kInfinity
     99     // is returned together with the computed isotropic safety
    100     // distance. Geometry must be closed.




    // g4-;g4-cls G4Transportation

    //   currentMinimumStep 
    //         winning minimum PhysicalStep from PostStepGPIL loop 

    163 G4double G4Transportation::
    164 AlongStepGetPhysicalInteractionLength( const G4Track&  track,
    165                                              G4double, //  previousStepSize
    166                                              G4double  currentMinimumStep,
    167                                              G4double& currentSafety,
    168                                              G4GPILSelection* selection )
    169 {
    170   G4double geometryStepLength= -1.0, newSafety= -1.0;
    ...
    265   if( !fieldExertsForce )
    266   {
    267      G4double linearStepLength ;
    268      if( fShortStepOptimisation && (currentMinimumStep <= currentSafety) )
    269      {
    270        // The Step is guaranteed to be taken
    271        //
    272        geometryStepLength   = currentMinimumStep ;
    273        fGeometryLimitedStep = false ;
    274      }
    275      else
    276      {
    277        //  Find whether the straight path intersects a volume
    278        //
    279        linearStepLength = fLinearNavigator->ComputeStep( startPosition,
    280                                                          startMomentumDir,
    281                                                          currentMinimumStep,
    282                                                          newSafety) ;
    283        // Remember last safety origin & value.
    284        //
    285        fPreviousSftOrigin = startPosition ;
    286        fPreviousSafety    = newSafety ;
    287        fpSafetyHelper->SetCurrentSafety( newSafety, startPosition);
    288 
    289        currentSafety = newSafety ;
    290      
    291        fGeometryLimitedStep= (linearStepLength <= currentMinimumStep);
    292        if( fGeometryLimitedStep )
    293        {
    294          // The geometry limits the Step size (an intersection was found.)
    295          geometryStepLength   = linearStepLength ;
    296        }
    297        else
    298        {
    299          // The full Step is taken.
    300          geometryStepLength   = currentMinimumStep ;
    301        }
    302      }
    303      fEndPointDistance = geometryStepLength ;
    304 
    305      // Calculate final position
    306      //
    307      fTransportEndPosition = startPosition+geometryStepLength*startMomentumDir ;
    308 
    309      // Momentum direction, energy and polarisation are unchanged by transport
    310      //
    311      fTransportEndMomentumDir   = startMomentumDir ;
    312      fTransportEndKineticEnergy = track.GetKineticEnergy() ;
    313      fTransportEndSpin          = track.GetPolarization();
    314      fParticleIsLooping         = false ;
    315      fMomentumChanged           = false ;
    316      fEndGlobalTimeComputed     = false ;
    317   }
    318   else   //  A field exerts force

