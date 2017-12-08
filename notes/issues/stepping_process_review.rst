stepping_process_review
=========================


Possible Geant4 no-scatter-no-scatter bug ? 
-------------------------------------------- 


The below method is central to G4 operation::

    095 void G4VProcess::ResetNumberOfInteractionLengthLeft()
     96 {
     97   theNumberOfInteractionLengthLeft =  -std::log( G4UniformRand() );
     98   theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft;
     99 }
    100 


Whilst attemping to align the random number consumption of 
Opticks and Geant4 optical only simulation I found that the
Geant4 equivalent ResetNumberOfInteractionLengthLeft to 
the below u_absorption and u_scattering throws was only being called 
for the first step of "TO BT BT SA".

The reason is that the PostStepDoItProc which invokes the base 
G4VDiscreteProcess::PostStepDoItProc only happens for the winning process.  

This means that the same scattering and absorption length probabilities are reused 
until those processes actually happen. For absoption thats the end of the line, so no-problem 
but for scattering it means that the probability of not scattering is artificially 
the same from step to step until a scatter actually happens. 

If this is a bug, I guess its a very minor one.


My workaround for this to align random consumption, invokes G4VDiscreteProcess::PostStepDoItProc 
at the end of the step for OpAbsorption and OpRayleigh in order to force interaction length 
resets for every step.



::

     59 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
     60 {
     61     //float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index    (phase velocity of light in medium)
     62     float speed = s.m1group2.x ;  // .x:group_velocity  (group velocity of light in the material) see: opticks-find GROUPVEL
     63 
     64 #ifdef WITH_ALIGN_DEV
     65 
     66     float u_boundary_burn = curand_uniform(&rng) ;
     67     float u_scattering = curand_uniform(&rng) ;
     68     float u_absorption = curand_uniform(&rng) ;
     69 
     70     float absorption_distance = -s.material1.y*logf(u_absorption) ;
     71     float scattering_distance = -s.material1.z*logf(u_scattering) ;
     72 
     73     rtPrintf("propagate_to_boundary  u_boundary_burn:%10.4f \n", u_boundary_burn );
     74     rtPrintf("propagate_to_boundary  u_scattering:%10.4f \n", u_scattering );
     75     rtPrintf("propagate_to_boundary  u_absorption:%10.4f \n", u_absorption );
     76 #else
     77     float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
     78     float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length
     79 #endif
     80 
     81     if (absorption_distance <= scattering_distance)
     82     {





lldb python scripted breakpoints
-----------------------------------

Setup scripted breakpoint dumping with lldb.

cfg4/g4lldb.py::

     41 def py_G4SteppingManager_DefinePhysicalStepLength(frame, bp_loc, sess):
     42     """
     43     ::
     44 
     45         g4-;g4-cls G4SteppingManager 
     46         g4-;g4-cls G4SteppingManager2
     47  
     48         tboolean-;tboolean-box --okg4 --align -D
     49 
     50 
     51         (lldb) b -f G4SteppingManager2.cc -l 181
     52 
     53             ## inside process loop after PostStepGPIL call giving physIntLength and fCondition
     54 
     55         (lldb) br com  add 1 -F opticks.cfg4.g4lldb.py_G4SteppingManager_DefinePhysicalStepLength 
     56 
     57     """
     58     name = sys._getframe().f_code.co_name
     59     
     60     global COUNT 
     61     COUNT[name] += 1
     62     
     63     kvar = "physIntLength fCondition PhysicalStep fStepStatus fPostStepDoItProcTriggered"
     64     
     65     this = frame.FindVariable("this")
     66     proc = this.GetChildMemberWithName("fCurrentProcess")
     67     procName = proc.GetChildMemberWithName("theProcessName")
     68     
     69     print 
     70     print FMT % ( name, COUNT[name] )
     71     print FMT % ( "procName", procName )
     72     
     73     for k in kvar.split():
     74         #v = frame.FindVariable(k)    gives no-value
     75         v = this.GetChildMemberWithName(k)
     76         print FMT % ( k, v )
     77     pass
     78     return False


Geant4 decision making : absorb/scatter/sail : absorb beats scatter but sail wins 
------------------------------------------------------------------------------------

Auto-breakpoint dumping from the above python

::

    //      py_G4SteppingManager_DefinePhysicalStepLength : 1 
    //                                           procName : (G4String) theProcessName = (std::__1::string = "OpBoundary") 
    //                                      physIntLength : (G4double) physIntLength = 1.7976931348623157E+308 
    //                                         fCondition : (G4ForceCondition) fCondition = Forced 
    //                                       PhysicalStep : (G4double) PhysicalStep = 1.7976931348623157E+308 
    //                                        fStepStatus : (G4StepStatus) fStepStatus = fUndefined 
    //                         fPostStepDoItProcTriggered : (size_t) fPostStepDoItProcTriggered = 4 

    //      py_G4SteppingManager_DefinePhysicalStepLength : 2 
    //                                           procName : (G4String) theProcessName = (std::__1::string = "OpRayleigh") 
    //                                      physIntLength : (G4double) physIntLength = 1004214.7797280541 
    //                                         fCondition : (G4ForceCondition) fCondition = NotForced 
    //                                       PhysicalStep : (G4double) PhysicalStep = 1.7976931348623157E+308 
    //                                        fStepStatus : (G4StepStatus) fStepStatus = fUndefined 
    //                         fPostStepDoItProcTriggered : (size_t) fPostStepDoItProcTriggered = 4 

    //      py_G4SteppingManager_DefinePhysicalStepLength : 3 
    //                                           procName : (G4String) theProcessName = (std::__1::string = "OpAbsorption") 
    //                                      physIntLength : (G4double) physIntLength = 587006.70078147366 
    //                                         fCondition : (G4ForceCondition) fCondition = NotForced 
    //                                       PhysicalStep : (G4double) PhysicalStep = 1004214.7797280541 
    //                                        fStepStatus : (G4StepStatus) fStepStatus = fPostStepDoItProc 
    //                         fPostStepDoItProcTriggered : (size_t) fPostStepDoItProcTriggered = 1 

    //      py_G4SteppingManager_DefinePhysicalStepLength : 4 
    //                                           procName : (G4String) theProcessName = (std::__1::string = "Transportation") 
    //                                      physIntLength : (G4double) physIntLength = 1.7976931348623157E+308 
    //                                         fCondition : (G4ForceCondition) fCondition = Forced 
    //                                       PhysicalStep : (G4double) PhysicalStep = 587006.70078147366 
    //                                        fStepStatus : (G4StepStatus) fStepStatus = fPostStepDoItProc 
    //                         fPostStepDoItProcTriggered : (size_t) fPostStepDoItProcTriggered = 2 




G4SteppingManager::DefinePhysicalStepLength
---------------------------------------------

Walk thru of below code makes sense, my problem
is why it doesnt happen the same way after the GeomBoundary  

* it has to happen, tis different material ...


As expected the below are both called 3 times for "TO BT BT SA"

::

   (lldb) b OpRayleigh::GetMeanFreePath   
   (lldb) b G4OpAbsorption::GetMeanFreePath


    (lldb) b G4VProcess::ResetNumberOfInteractionLengthLeft



::

    g4-;g4-cls G4SteppingManager
    g4-;g4-cls G4SteppingManager2

    G4SteppingManager::DefinePhysicalStepLength

    127 /////////////////////////////////////////////////////////
    128  void G4SteppingManager::DefinePhysicalStepLength()
    129 /////////////////////////////////////////////////////////
    130 {
    131 
    132 // ReSet the counter etc.
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
    ...
     

    (lldb) b G4SteppingManager::DefinePhysicalStepLength
    (lldb) r
    (lldb) b 181
    (lldb) b 270  # for summary
    (lldb) c

    (lldb) frame variable fCurrentProcess->theProcessName physIntLength fCondition PhysicalStep

    (G4String) fCurrentProcess->theProcessName = (std::__1::string = "OpRayleigh")
    (G4double) physIntLength = 58700.67007814737
    (G4ForceCondition) fCondition = NotForced
    (G4double) PhysicalStep = 1.7976931348623157E+308

    (lldb) p (double)log(0.942989)*-1e6
    (double) $1 = 58700.661315972749


    (lldb) frame variable fCurrentProcess->theProcessName physIntLength fCondition PhysicalStep fStepStatus fPostStepDoItProcTriggered

    (G4String) fCurrentProcess->theProcessName = (std::__1::string = "OpAbsorption")
    (G4double) physIntLength = 12766112.786981029
    (G4ForceCondition) fCondition = NotForced
    (G4double) PhysicalStep = 58700.67007814737
    (G4StepStatus) fStepStatus = fPostStepDoItProc
    (size_t) fPostStepDoItProcTriggered = 2

    ## OpRayleigh in lead 

    (lldb) p (double)log(0.278981)*-1e6
    (double) $2 = 1276611.599838129

    (lldb) p (double)log(0.278981)*-1e7
    (double) $3 = 12766115.998381291


    181      switch (fCondition) {
        182      case ExclusivelyForced:
        183          (*fSelectedPostStepDoItVector)[np] = ExclusivelyForced;
        184          fStepStatus = fExclusivelyForcedProc;
        185          fStep->GetPostStepPoint()
        186          ->SetProcessDefinedStep(fCurrentProcess);
        187          break;
        ...
        193      case Forced:
        194          (*fSelectedPostStepDoItVector)[np] = Forced;
        195          break;
        196      case StronglyForced:
        197          (*fSelectedPostStepDoItVector)[np] = StronglyForced;
        198          break;
        199      default:
        200          (*fSelectedPostStepDoItVector)[np] = InActivated;
        ////    ^^^^^^^^^  hmm NotForced gets InActivated, have to set some condition to stay selected 
        201          break;
    202      }


    (lldb) b G4SteppingManager::DefinePhysicalStepLength
    (lldb) b 206




G4SteppingManager::DefinePhysicalStepLength  are proceeses being nullified ? : NO
-------------------------------------------------------------------------------------

::

    (lldb) fr v *fPostStepGetPhysIntVector->pProcVector
    (G4ProcessVector::G4ProcVector) *fPostStepGetPhysIntVector->pProcVector = size=5 {
      [0] = 0x000000010f7a7030
      [1] = 0x000000010f7a8f00
      [2] = 0x000000010f7a8d70
      [3] = 0x000000010f7a8770
      [4] = 0x000000010f77fd70




SetProcessDefinedStep for the winning process
-----------------------------------------------

::



    206      if (fCondition==ExclusivelyForced) {
    207          for(size_t nrest=np+1; nrest < MAXofPostStepLoops; nrest++){
    208              (*fSelectedPostStepDoItVector)[nrest] = InActivated;
    209          }
    210          return;  // Take note the 'return' at here !!! 
    211      }
    212      else{
    213          if(physIntLength < PhysicalStep ){
    214              PhysicalStep = physIntLength;
    215              fStepStatus = fPostStepDoItProc;
    216              fPostStepDoItProcTriggered = G4int(np);
    217              fStep->GetPostStepPoint()
    218                  ->SetProcessDefinedStep(fCurrentProcess);
    219          }
    220      }
    223    }



    225    if (fPostStepDoItProcTriggered<MAXofPostStepLoops) {
    226        if ((*fSelectedPostStepDoItVector)[fPostStepDoItProcTriggered] ==
    227        InActivated) {
    228        (*fSelectedPostStepDoItVector)[fPostStepDoItProcTriggered] =
    229            NotForced;
    230        }
    231    }

::

    (lldb) p *fAlongStepGetPhysIntVector
    (G4ProcessVector) $6 = {
      pProcVector = 0x0000000111144560 size=1
    }



AlongStepGPIL Process Loop : often just Transportation
---------------------------------------------------------

* G4VDiscreteProcess just does Post, no Along or AtRest
  so this will usually be just Transportation with optical photons
  (what about Scint ?)

::

    (lldb) b 251


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

    ///  PhysicalStep here comes from above np loop

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



    (lldb) b 270
    lldb) frame variable fStepStatus MAXofAlongStepLoops fGPILSelection physIntLength PhysicalStep safetyProposedToAndByProcess
    (G4StepStatus) fStepStatus = fGeomBoundary
    (size_t) MAXofAlongStepLoops = 1
    (G4GPILSelection) fGPILSelection = CandidateForSelection
    (G4double) physIntLength = 349.89999389648438
    (G4double) PhysicalStep = 349.89999389648438
    (G4double) safetyProposedToAndByProcess = 0.100006103515625
    (lldb) 



    270      if (safetyProposedToAndByProcess < proposedSafety)
    271         // proposedSafety keeps the smallest value:
    272         proposedSafety               = safetyProposedToAndByProcess;
    273      else
    274         // safetyProposedToAndByProcess always proposes a valid safety:
    275         safetyProposedToAndByProcess = proposedSafety;
    276      
    277    }
    278 } // void G4SteppingManager::DefinePhysicalStepLength() //


::

    (lldb) frame var  fStep->fpPreStepPoint->fPosition fStep->fpPreStepPoint->fGlobalTime fStep->fpPreStepPoint->fMomentumDirection  fStep->fpPreStepPoint->fpMaterial->fName
    (G4ThreeVector) fStep->fpPreStepPoint->fPosition = (dx = 11.291412353515625, dy = -34.645111083984375, dz = -449.89999389648438)
    (G4double) fStep->fpPreStepPoint->fGlobalTime = 0.20000000298023224
    (G4ThreeVector) fStep->fpPreStepPoint->fMomentumDirection = (dx = -0, dy = -0, dz = 1)
    (G4String) fStep->fpPreStepPoint->fpMaterial->fName = (std::__1::string = "Vacuum")
    (lldb) 



    (lldb) frame variable fStepStatus MAXofAlongStepLoops fGPILSelection physIntLength PhysicalStep safetyProposedToAndByProcess
    (G4StepStatus) fStepStatus = fGeomBoundary
    (size_t) MAXofAlongStepLoops = 1
    (G4GPILSelection) fGPILSelection = CandidateForSelection
    (G4double) physIntLength = 200
    (G4double) PhysicalStep = 200
    (G4double) safetyProposedToAndByProcess = 0
    (lldb) frame var  fStep->fpPreStepPoint->fPosition fStep->fpPreStepPoint->fGlobalTime fStep->fpPreStepPoint->fMomentumDirection  fStep->fpPreStepPoint->fpMaterial->fName
    (G4ThreeVector) fStep->fpPreStepPoint->fPosition = (dx = 11.291412353515625, dy = -34.645111083984375, dz = -100)
    (G4double) fStep->fpPreStepPoint->fGlobalTime = 1.3671407830548261
    (G4ThreeVector) fStep->fpPreStepPoint->fMomentumDirection = (dx = -0, dy = -0, dz = 1)
    (G4String) fStep->fpPreStepPoint->fpMaterial->fName = (std::__1::string = "GlassSchottF2")
    (lldb) 


    (lldb) frame var  fStep->fpPreStepPoint->fPosition fStep->fpPreStepPoint->fGlobalTime fStep->fpPreStepPoint->fMomentumDirection  fStep->fpPreStepPoint->fpMaterial->fName
    (G4ThreeVector) fStep->fpPreStepPoint->fPosition = (dx = 11.291412353515625, dy = -34.645111083984375, dz = 100)
    (G4double) fStep->fpPreStepPoint->fGlobalTime = 2.5790558894519888
    (G4ThreeVector) fStep->fpPreStepPoint->fMomentumDirection = (dx = -0, dy = -0, dz = 1)
    (G4String) fStep->fpPreStepPoint->fpMaterial->fName = (std::__1::string = "Vacuum")
    (lldb) 

    (lldb) frame variable fStepStatus MAXofAlongStepLoops fGPILSelection physIntLength PhysicalStep safetyProposedToAndByProcess
    (G4StepStatus) fStepStatus = fGeomBoundary
    (size_t) MAXofAlongStepLoops = 1
    (G4GPILSelection) fGPILSelection = CandidateForSelection
    (G4double) physIntLength = 350
    (G4double) PhysicalStep = 350
    (G4double) safetyProposedToAndByProcess = 0
    (lldb) 






G4SteppingManager::InvokePostStepDoItProcs
-------------------------------------------

G4VDiscreteProcess::PostStepDoIt which clears interaction lengths
G4VProcess::ClearNumberOfInteractionLengthLeft is only called for OpBoundary 

* why ?

::

    483 void G4SteppingManager::InvokePostStepDoItProcs()
    484 ////////////////////////////////////////////////////////
    485 {
    486 
    487 // Invoke the specified discrete processes
    488    for(size_t np=0; np < MAXofPostStepLoops; np++){
    489    //
    490    // Note: DoItVector has inverse order against GetPhysIntVector
    491    //       and SelectedPostStepDoItVector.
    492    //
    493      G4int Cond = (*fSelectedPostStepDoItVector)[MAXofPostStepLoops-np-1];
    494      if(Cond != InActivated){
    495        if( ((Cond == NotForced) && (fStepStatus == fPostStepDoItProc)) ||
    496            ((Cond == Forced) && (fStepStatus != fExclusivelyForcedProc)) ||
    498            ((Cond == ExclusivelyForced) && (fStepStatus == fExclusivelyForcedProc)) ||
    499            ((Cond == StronglyForced) )
    500       ) {
    501 
    502          InvokePSDIP(np);
    503          if ((np==0) && (fTrack->GetNextVolume() == 0)){
    504            fStepStatus = fWorldBoundary;
    505            fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
    506          }
    507        }
    508      } //if(*fSelectedPostStepDoItVector(np)........
    509 
    510      // Exit from PostStepLoop if the track has been killed,
    511      // but extra treatment for processes with Strongly Forced flag
    512      if(fTrack->GetTrackStatus() == fStopAndKill) {
    513        for(size_t np1=np+1; np1 < MAXofPostStepLoops; np1++){
    514            G4int Cond2 = (*fSelectedPostStepDoItVector)[MAXofPostStepLoops-np1-1];
    515            if (Cond2 == StronglyForced) {
    516                InvokePSDIP(np1);
    517            }
    518        }
    519        break;
    520      }
    521    } //for(size_t np=0; np < MAXofPostStepLoops; np++){
    522 }




G4TrackingManager : big picture wrt stepping
-------------------------------------------------

::

    g4-;g4-cls G4TrackingManager


    067 void G4TrackingManager::ProcessOneTrack(G4Track* apValueG4Track)
     69 {
     71   // Receiving a G4Track from the EventManager, this funciton has the
     72   // responsibility to trace the track till it stops.
     73   fpTrack = apValueG4Track;
     74   EventIsAborted = false;
    ...
     88   // Give SteppingManger the pointer to the track which will be tracked 
     89   fpSteppingManager->SetInitialStep(fpTrack);
     90 
     91   // Pre tracking user intervention process.
     93   if( fpUserTrackingAction != 0 ) {
     94      fpUserTrackingAction->PreUserTrackingAction(fpTrack);
     95   }
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







