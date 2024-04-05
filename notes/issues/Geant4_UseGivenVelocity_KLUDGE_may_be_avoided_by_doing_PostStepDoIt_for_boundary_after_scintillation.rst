Geant4_UseGivenVelocity_KLUDGE_may_be_avoided_by_doing_PostStepDoIt_for_boundary_after_scintillation
======================================================================================================

Where G4ParticleChange::ProposeVelocity is called from 
----------------------------------------------------------

::

    BP=G4ParticleChange::ProposeVelocity  ~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh
    BP=G4ParticleChange::ProposeVelocity ~/o/g4cx/tests/G4CXTest_raindrop_CPU.sh 


Checking where ProposeVelocity is called in pure optical test,
only see head and tail of G4OpBoundaryProcess::PostStepDoIt.

::

    lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x000000010214fad1 libG4processes.dylib`G4ParticleChange::ProposeVelocity(this=0x000000010714dae8, finalVelocity=299.79245800000001) at G4ParticleChange.icc:57
        frame #1: 0x0000000103541e2b libG4processes.dylib`G4OpBoundaryProcess::PostStepDoIt(this=0x000000010714dad0, aTrack=0x0000000114262c80, aStep=0x000000010712b1c0) at G4OpBoundaryProcess.cc:546
         ....

        frame #10: 0x0000000101b7ecd1 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010712ad80, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #11: 0x0000000100005c63 OpticalAppTest`OpticalApp::Main() at OpticalApp.h:90
        frame #12: 0x0000000100007e04 OpticalAppTest`main at OpticalAppTest.cc:4
        frame #13: 0x00007fff50548015 libdyld.dylib`start + 1
    (lldb) p theStatus
    error: use of undeclared identifier 'theStatus'
    (lldb) f 1
    frame #1: 0x0000000103541e2b libG4processes.dylib`G4OpBoundaryProcess::PostStepDoIt(this=0x000000010714dad0, aTrack=0x0000000114262c80, aStep=0x000000010712b1c0) at G4OpBoundaryProcess.cc:546
       543 	           G4MaterialPropertyVector* groupvel =
       544 	           Material2->GetMaterialPropertiesTable()->GetProperty(kGROUPVEL);
       545 	           G4double finalVelocity = groupvel->Value(thePhotonMomentum);
    -> 546 	           aParticleChange.ProposeVelocity(finalVelocity);
       547 	        }
       548 	
       549 	        if ( theStatus == Detection && fInvokeSD ) InvokeSD(pStep);
    (lldb) p theStatus
    (G4OpBoundaryProcessStatus) $0 = FresnelRefraction
    (lldb) 



HUH: 1042 G4CXTest_raindrop.sh needs kludge to get expected by OpticalApp does not, what gives ? 
-----------------------------------------------------------------------------------------------------

raindrop without kludge, goes wrong::

    [[b'TO BR BR BR BR BR BR BR BT SA                                                                   ' b'2']
     [b'TO BR BR BR BR BR BR BT SA                                                                      ' b'12']
     [b'TO BR BR BR BR BR BT SA                                                                         ' b'45']
     [b'TO BR BR BR BR BT SA                                                                            ' b'109']
     [b'TO BR BR BR BT SA                                                                               ' b'880']
     [b'TO BR BR BT SA                                                                                  ' b'2465']
     [b'TO BR BT SA                                                                                     ' b'46578']
     [b'TO BT SA                                                                                        ' b'49909']]
    PICK=B MODE=3 SELECT="TO BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed len/min/max for : 0 -> 1 : TO -> BT :    49909 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BT -> SA :    49909 224.901 224.901 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1042 
    _pos.shape (49909, 3) 

    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed len/min/max for : 0 -> 1 : TO -> BR :    46578 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BR -> BT :    46578 299.792 299.793 
    speed len/min/max for : 2 -> 3 : BT -> SA :    46578 224.901 224.901 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1042 
    _pos.shape (46578, 3) 


raindrop With kludge, get expected::

    np.c_[np.unique(b.q, return_counts=True)] 
    [[b'TO BR BR BR BR BR BR BR BT SA                                                                   ' b'2']
     [b'TO BR BR BR BR BR BR BT SA                                                                      ' b'12']
     [b'TO BR BR BR BR BR BT SA                                                                         ' b'45']
     [b'TO BR BR BR BR BT SA                                                                            ' b'109']
     [b'TO BR BR BR BT SA                                                                               ' b'880']
     [b'TO BR BR BT SA                                                                                  ' b'2465']
     [b'TO BR BT SA                                                                                     ' b'46578']
     [b'TO BT SA                                                                                        ' b'49909']]
    PICK=B MODE=3 SELECT="TO BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed len/min/max for : 0 -> 1 : TO -> BT :    49909 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BT -> SA :    49909 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:1 
    e.f.NPFold_meta.G4VERSION_NUMBER:1042 
    _pos.shape (49909, 3) 
    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed len/min/max for : 0 -> 1 : TO -> BR :    46578 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BR -> BT :    46578 224.901 224.901 
    speed len/min/max for : 2 -> 3 : BT -> SA :    46578 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:1 
    e.f.NPFold_meta.G4VERSION_NUMBER:1042 
    _pos.shape (46578, 3) 



OpticalApp without, correct::

    [[b'TO BR BR BR BR BR BR BR BT SA                                                                   ' b'3']
     [b'TO BR BR BR BR BR BR BT SA                                                                      ' b'11']
     [b'TO BR BR BR BR BR BT SA                                                                         ' b'38']
     [b'TO BR BR BR BR BT SA                                                                            ' b'105']
     [b'TO BR BR BR BT SA                                                                               ' b'868']
     [b'TO BR BR BT SA                                                                                  ' b'2419']
     [b'TO BR BT SA                                                                                     ' b'46620']
     [b'TO BT SA                                                                                        ' b'49936']]
    SELECT="TO BT SA" ~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh
    speed len/min/max for : 0 -> 1 : TO -> BT :   49936/224.901/224.901 
    speed len/min/max for : 1 -> 2 : BT -> SA :   49936/299.792/299.792 
    source:OpticalApp::desc
    OpticalApp__PreUserTrackingAction_UseGivenVelocity_KLUDGE:0
    SELECT="TO BR BT SA" ~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh
    speed len/min/max for : 0 -> 1 : TO -> BR :   46620/224.901/224.901 
    speed len/min/max for : 1 -> 2 : BR -> BT :   46620/224.901/224.901 
    speed len/min/max for : 2 -> 3 : BT -> SA :   46620/299.792/299.792 
    source:OpticalApp::desc
    OpticalApp__PreUserTrackingAction_UseGivenVelocity_KLUDGE:0


With kludge, makes no difference for OpticalApp, correct either way::

    np.c_[np.unique(b.q, return_counts=True)] 
    [[b'TO BR BR BR BR BR BR BR BT SA                                                                   ' b'3']
     [b'TO BR BR BR BR BR BR BT SA                                                                      ' b'11']
     [b'TO BR BR BR BR BR BT SA                                                                         ' b'38']
     [b'TO BR BR BR BR BT SA                                                                            ' b'105']
     [b'TO BR BR BR BT SA                                                                               ' b'868']
     [b'TO BR BR BT SA                                                                                  ' b'2419']
     [b'TO BR BT SA                                                                                     ' b'46620']
     [b'TO BT SA                                                                                        ' b'49936']]
    SELECT="TO BT SA" ~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh
    speed len/min/max for : 0 -> 1 : TO -> BT :   49936/224.901/224.901 
    speed len/min/max for : 1 -> 2 : BT -> SA :   49936/299.792/299.792 
    source:OpticalApp::desc
    OpticalApp__PreUserTrackingAction_UseGivenVelocity_KLUDGE:1
    SELECT="TO BR BT SA" ~/o/examples/Geant4/OpticalApp/OpticalAppTest.sh
    speed len/min/max for : 0 -> 1 : TO -> BR :   46620/224.901/224.901 
    speed len/min/max for : 1 -> 2 : BR -> BT :   46620/224.901/224.901 
    speed len/min/max for : 2 -> 3 : BT -> SA :   46620/299.792/299.792 
    source:OpticalApp::desc
    OpticalApp__PreUserTrackingAction_UseGivenVelocity_KLUDGE:1



Use VERSION 0/1 to separate without/with KLUDGE records and compare animations
---------------------------------------------------------------------------------

These makes it plain that it is only time that is effected, the refraction angle is the same in both cases::

    REC=/data/blyth/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000/TO_BT_SA SHADER=pos ~/o/examples/UseGeometryShader/run.sh
    REC=/data/blyth/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL1/B000/TO_BT_SA SHADER=pos ~/o/examples/UseGeometryShader/run.sh


Debug
------

::

    BP=G4Track::GetVelocity ~/opticks/g4cx/tests/G4CXTest_raindrop.sh

    BP=G4StepPoint::GetVelocity  ~/opticks/g4cx/tests/G4CXTest_raindrop.sh

::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.2
      * frame #0: 0x00000001026cc84c libG4processes.dylib`G4Track::GetVelocity(this=0x00000001172ee5f0) const at G4Track.icc:121
        frame #1: 0x00000001026cd16d libG4processes.dylib`G4ParticleChangeForTransport::Initialize(this=0x000000010ac3a4f0, track=0x00000001172ee5f0) at G4ParticleChangeForTransport.icc:109
        frame #2: 0x0000000103b203c5 libG4processes.dylib`G4Transportation::AlongStepDoIt(this=0x000000010ac3a2d0, track=0x00000001172ee5f0, stepData=0x000000010aafbf40) at G4Transportation.cc:525
        frame #3: 0x0000000102317eb9 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x000000010ac00000) at G4SteppingManager2.cc:421   

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.2
      * frame #0: 0x00000001026cc84c libG4processes.dylib`G4Track::GetVelocity(this=0x00000001172ee5f0) const at G4Track.icc:121
        frame #1: 0x0000000105372e81 libG4track.dylib`G4ParticleChange::Initialize(this=0x000000010ac6eac8, track=0x00000001172ee5f0) at G4ParticleChange.cc:237
        frame #2: 0x00000001065a400c libCustom4.dylib`C4OpBoundaryProcess::PostStepDoIt(this=0x000000010ac6eab0, aTrack=0x00000001172ee5f0, aStep=0x000000010aafbf40) at C4OpBoundaryProcess.cc:211
        frame #3: 0x00000001023187db libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x000000010ac00000, np=3) at G4SteppingManager2.cc:538
        frame #4: 0x000000010231864d libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x000000010ac00000) at G4SteppingManager2.cc:510

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.2
      * frame #0: 0x00000001026cc84c libG4processes.dylib`G4Track::GetVelocity(this=0x00000001172ee5f0) const at G4Track.icc:121
        frame #1: 0x0000000105372e81 libG4track.dylib`G4ParticleChange::Initialize(this=0x000000010ac6db58, track=0x00000001172ee5f0) at G4ParticleChange.cc:237
        frame #2: 0x0000000100543172 libU4.dylib`Local_DsG4Scintillation::PostStepDoIt(this=0x000000010ac6db40, aTrack=0x00000001172ee5f0, aStep=0x000000010aafbf40) at Local_DsG4Scintillation.cc:264
        frame #3: 0x00000001023187db libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x000000010ac00000, np=4) at G4SteppingManager2.cc:538
        frame #4: 0x000000010231864d libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x000000010ac00000) at G4SteppingManager2.cc:510

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.2
      * frame #0: 0x00000001026cc84c libG4processes.dylib`G4Track::GetVelocity(this=0x00000001172ee5f0) const at G4Track.icc:121
        frame #1: 0x00000001026cd16d libG4processes.dylib`G4ParticleChangeForTransport::Initialize(this=0x000000010ac3a4f0, track=0x00000001172ee5f0) at G4ParticleChangeForTransport.icc:109
        frame #2: 0x0000000103b203c5 libG4processes.dylib`G4Transportation::AlongStepDoIt(this=0x000000010ac3a2d0, track=0x00000001172ee5f0, stepData=0x000000010aafbf40) at G4Transportation.cc:525
        frame #3: 0x0000000102317eb9 libG4tracking.dylib`G4SteppingManager::InvokeAlongStepDoItProcs(this=0x000000010ac00000) at G4SteppingManager2.cc:421



g4-cls G4SteppingManager2
---------------------------

::

    534 void G4SteppingManager::InvokePSDIP(size_t np)
    535 {
    536          fCurrentProcess = (*fPostStepDoItVector)[np];
    537          fParticleChange
    538             = fCurrentProcess->PostStepDoIt( *fTrack, *fStep);
    539 
    540          // Update PostStepPoint of Step according to ParticleChange
    541      fParticleChange->UpdateStepForPostStep(fStep);
    542 #ifdef G4VERBOSE
    543                  // !!!!! Verbose
    544            if(verboseLevel>0) fVerbose->PostStepDoItOneByOne();
    545 #endif
    546          // Update G4Track according to ParticleChange after each PostStepDoIt
    547          fStep->UpdateTrack();
    548 
    549          // Update safety after each invocation of PostStepDoIts
    550          fStep->GetPostStepPoint()->SetSafety( CalculateSafety() );
    551 
    552          // Now Store the secondaries from ParticleChange to SecondaryList
    553          G4Track* tempSecondaryTrack;
    554          G4int    num2ndaries;
    555 
    556          num2ndaries = fParticleChange->GetNumberOfSecondaries();




HMM : maybe reemission handling messes the velocity ?
--------------------------------------------------------

::

     255 G4VParticleChange*
     256 Local_DsG4Scintillation::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     257 
     258 // This routine is called for each tracking step of a charged particle
     259 // in a scintillator. A Poisson/Gauss-distributed number of photons is 
     260 // generated according to the scintillation yield formula, distributed 
     261 // evenly along the track segment and uniformly into 4pi.
     262 
     263 {
     264     aParticleChange.Initialize(aTrack);
     265 
     266     if (m_noop) {               // do nothing, bail
     267         aParticleChange.SetNumberOfSecondaries(0);
     268         return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
     269     }
     270 


::

    BP="G4Track::GetVelocity G4ParticleChange::ProposeVelocity"  ~/opticks/g4cx/tests/G4CXTest_raindrop.sh     


::

    519 G4VParticleChange* G4Transportation::AlongStepDoIt( const G4Track& track,
    520                                                     const G4Step&  stepData )
    521 {
    522   static G4ThreadLocal G4int noCalls=0;
    523   noCalls++;
    524 
    525   fParticleChange.Initialize(track) ;
    526 
    527   //  Code for specific process 
    528   //
    529   fParticleChange.ProposePosition(fTransportEndPosition) ;
    530   fParticleChange.ProposeMomentumDirection(fTransportEndMomentumDir) ;
    531   fParticleChange.ProposeEnergy(fTransportEndKineticEnergy) ;
    532   fParticleChange.SetMomentumChanged(fMomentumChanged) ;
    533 
    534   fParticleChange.ProposePolarization(fTransportEndSpin);
    535 
    536   G4double deltaTime = 0.0 ;
    537 
    538   // Calculate  Lab Time of Flight (ONLY if field Equations used it!)
    539   // G4double endTime   = fCandidateEndGlobalTime;
    540   // G4double delta_time = endTime - startTime;
    541 
    542   G4double startTime = track.GetGlobalTime() ;
    543  
    544   if (!fEndGlobalTimeComputed)
    545   {  
    546      // The time was not integrated .. make the best estimate possible
    547      //
    548      G4double initialVelocity = stepData.GetPreStepPoint()->GetVelocity();
    549      G4double stepLength      = track.GetStepLength();
    550 
    551      deltaTime= 0.0;  // in case initialVelocity = 0 
    552      if ( initialVelocity > 0.0 )  { deltaTime = stepLength/initialVelocity; }
    553 
    554      fCandidateEndGlobalTime   = startTime + deltaTime ;
    555      fParticleChange.ProposeLocalTime(  track.GetLocalTime() + deltaTime) ;
    556   }
    557   else
    558   {
    559      deltaTime = fCandidateEndGlobalTime - startTime ;
    560      fParticleChange.ProposeGlobalTime( fCandidateEndGlobalTime ) ;
    561   }


The velocity that matters is G4StepPoint::GetVelocity 

::

    BP=G4StepPoint::GetVelocity 



BINGO : disabling scintillation which is there for reemission prevents need for the kludge
--------------------------------------------------------------------------------------------

::
 
    export Local_DsG4Scintillation_DISABLE=1



::

    np.c_[np.unique(b.q, return_counts=True)] 
    [[b'TO BR BR BR BR BR BR BR BR BT SA                                                                ' b'2']
     [b'TO BR BR BR BR BR BR BT SA                                                                      ' b'9']
     [b'TO BR BR BR BR BR BT SA                                                                         ' b'43']
     [b'TO BR BR BR BR BT SA                                                                            ' b'144']
     [b'TO BR BR BR BT SA                                                                               ' b'819']
     [b'TO BR BR BT SA                                                                                  ' b'2461']
     [b'TO BR BT SA                                                                                     ' b'46617']
     [b'TO BT SA                                                                                        ' b'49905']]
    PICK=B MODE=3 SELECT="TO BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/data/blyth/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000/TO_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BT :    49905 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BT -> SA :    49905 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1042 
    _pos.shape (49905, 3) 
    _beg.shape (49905, 3) 
    _poi.shape (49905, 3, 3) 
    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    REC=/data/blyth/opticks/GEOM/RaindropRockAirWater/G4CXTest/ALL0/B000/TO_BR_BT_SA ~/o/examples/UseGeometryShader/run.sh
    speed len/min/max for : 0 -> 1 : TO -> BR :    46617 224.901 224.901 
    speed len/min/max for : 1 -> 2 : BR -> BT :    46617 224.901 224.901 
    speed len/min/max for : 2 -> 3 : BT -> SA :    46617 299.792 299.793 
    e.f.NPFold_meta.U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0 
    e.f.NPFold_meta.G4VERSION_NUMBER:1042 
    _pos.shape (46617, 3) 
    _beg.shape (46617, 3) 
    _poi.shape (46617, 4, 3) 



::

    BP=Local_DsG4Scintillation::PostStepDoIt ~/opticks/g4cx/tests/G4CXTest_raindrop.sh


Called twice with "TO BT SA" 

::

    (lldb) f 1
    frame #1: 0x00000001023142e5 libG4tracking.dylib`G4Step::UpdateTrack(this=0x000000010abca860) at G4Step.icc:251
       248 	
       249 	
       250 	   // set velocity 
    -> 251 	   fpTrack->SetVelocity(fpPostStepPoint->GetVelocity());
       252 	}
       253 	
       254 	inline  G4int G4Step::GetNumberOfSecondariesInCurrentStep() const
    (lldb) 



10 G4Track::SetVelocity hits for "TO BT SA" 4 and 10 are 299
----------------------------------------------------------------

::

    epsilon:issues blyth$ BP=G4Track::SetVelocity  ~/opticks/g4cx/tests/G4CXTest_raindrop.sh



The supposedly do nothing Local_DsScintillation called after C4OpBoundaryProcess trumping the correct velocity::

    arget 0: (G4CXTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000102315581 libG4tracking.dylib`G4Track::SetVelocity(this=0x000000010ad1aa80, val=224.90056864216055) at G4Track.icc:124
        frame #1: 0x00000001023142ee libG4tracking.dylib`G4Step::UpdateTrack(this=0x000000010ac89c30) at G4Step.icc:251
        frame #2: 0x0000000102318859 libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x000000010ac89aa0, np=4) at G4SteppingManager2.cc:547
        frame #3: 0x000000010231864d libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x000000010ac89aa0) at G4SteppingManager2.cc:510
        frame #4: 0x0000000102313daa libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010ac89aa0) at G4SteppingManager.cc:209
        frame #5: 0x000000010232a86f libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010ac89a60, apValueG4Track=0x000000010ad1aa80) at G4TrackingManager.cc:126
        frame #6: 0x00000001021f071a libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010ac899d0, anEvent=0x000000010de6c150) at G4EventManager.cc:185
        frame #7: 0x00000001021f1c2f libG4event.dylib`G4EventManager::ProcessOneEvent(this=0x000000010ac899d0, anEvent=0x000000010de6c150) at G4EventManager.cc:338
        frame #8: 0x00000001020fd9e5 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010ac897f0, i_event=0) at G4RunManager.cc:399
        frame #9: 0x00000001020fd815 libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010ac897f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
        frame #10: 0x00000001020fbcd1 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010ac897f0, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
        frame #11: 0x00000001000560af G4CXTest`G4CXApp::BeamOn(this=0x000000010acb50d0) at G4CXApp.h:344
        frame #12: 0x00000001000561da G4CXTest`G4CXApp::Main() at G4CXApp.h:351
        frame #13: 0x000000010005640c G4CXTest`main(argc=1, argv=0x00007ffeefbfe578) at G4CXTest.cc:13
        frame #14: 0x00007fff50548015 libdyld.dylib`start + 1
        frame #15: 0x00007fff50548015 libdyld.dylib`start + 1
    (lldb) f 2
    frame #2: 0x0000000102318859 libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x000000010ac89aa0, np=4) at G4SteppingManager2.cc:547
       544 	           if(verboseLevel>0) fVerbose->PostStepDoItOneByOne();
       545 	#endif
       546 	         // Update G4Track according to ParticleChange after each PostStepDoIt
    -> 547 	         fStep->UpdateTrack();
       548 	
       549 	         // Update safety after each invocation of PostStepDoIts
       550 	         fStep->GetPostStepPoint()->SetSafety( CalculateSafety() );
    (lldb) p fCurrentProcess
    (Local_DsG4Scintillation *) $7 = 0x000000010aa610e0
    (lldb) 


HMM: maybe doing boundary after scint ?
-------------------------------------------

::

    BP=G4VProcess::PostStepDoIt ~/opticks/g4cx/tests/G4CXTest_raindrop.sh

     


AtRestDoIt calls PostStepDoIt for scint
-----------------------------------------

::

     242 G4VParticleChange*
     243 Local_DsG4Scintillation::AtRestDoIt(const G4Track& aTrack, const G4Step& aStep)
     244 
     245 // This routine simply calls the equivalent PostStepDoIt since all the
     246 // necessary information resides in aStep.GetTotalEnergyDeposit()
     247 
     248 {
     249     return Local_DsG4Scintillation::PostStepDoIt(aTrack, aStep);
     250 }




LASTPOST seems to work but get warning
---------------------------------------

::

    ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    ...
  
    2024-04-05 15:40:50.581 INFO  [17694864] [U4Physics::ConstructOp@251] U4Physics::desc
                         U4Physics__ConstructOp_Cerenkov_DISABLE : 0
                    U4Physics__ConstructOp_Scintillation_DISABLE : 0
                     U4Physics__ConstructOp_OpAbsorption_DISABLE : 0
                       U4Physics__ConstructOp_OpRayleigh_DISABLE : 0
                U4Physics__ConstructOp_OpBoundaryProcess_DISABLE : 0
               U4Physics__ConstructOp_OpBoundaryProcess_LASTPOST : 1
                           U4Physics__ConstructOp_FastSim_ENABLE : 0

    Local_DsG4Scintillation::Local_DsG4Scintillation level 0 verboseLevel 0
    2024-04-05 15:40:50.581 FATAL [17694864] [*U4Physics::CreateBoundaryProcess@371]  FAILED TO SPMTAccessor::Load from [$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/extra/jpmt] GEOM RaindropRockAirWater
    2024-04-05 15:40:50.581 INFO  [17694864] [U4Physics::ConstructOp@294]  fBoundary 0x7f9007cb8c90

    -------- WWWW ------- G4Exception-START -------- WWWW -------
    *** G4Exception : ProcMan114
          issued by : G4ProcessManager::SetProcessOrderingToLast()
    Set Ordering Last is invoked twice for OpBoundary to opticalphoton
    *** This is just a warning message. ***
    -------- WWWW -------- G4Exception-END --------- WWWW -------

    2024-04-05 15:40:50.583 INFO  [17694864] [G4CXApp::G4CXApp@160] 
    U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0


::

    BP=G4ProcessManager::SetProcessOrderingToLast ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 




