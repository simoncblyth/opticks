reemission review
=====================


CRecorder::Record assuming reemission "secondary"  tracks immediately after the AB tracks
------------------------------------------------------------------------------------------

::

    090 /**
     91 CRecorder::Record
     92 ===================
     93 
     94 Not-zeroing m_slot for REJOINders 
     95 ----------------------------------
     96 
     97 * see notes/issues/reemission_review.rst
     98 
     99 Rejoining happens on output side not in the crec CStp list.
    100 
    101 The rejoins of AB(actually RE) tracks with reborn secondaries 
    102 are done by writing two (or more) sequencts of track steps  
    103 into the same record_id in the record buffer at the 
    104 appropiate non-zeroed slot.
    105 
    106 WAS a bit confused by this ...
    107  
    108 This assumes that the REJOINing track will
    109 be the one immediately after the original AB. 
    110 By virtue of the Cerenkov/Scintillation process setting:
    111 
    112      SetTrackSecondariesFirst(true)
    113   
    114 If not so, this will "join" unrelated tracks ?
    115 
    116 Does this mean the local photon state is just for live mode ?
    117 
    118 
    119 **/
    120 
    121 // invoked by CSteppingAction::setStep
    122 #ifdef USE_CUSTOM_BOUNDARY
    123 bool CRecorder::Record(DsG4OpBoundaryProcessStatus boundary_status)
    124 #else
    125 bool CRecorder::Record(G4OpBoundaryProcessStatus boundary_status)
    126 #endif
    127 {    
    128     m_step_action = 0 ;
    129 
    130     if(m_ctx._dbgrec)
    131     LOG(trace) << "CRecorder::Record"
    132               << " step_id " << m_ctx._step_id
    133               << " record_id " << m_ctx._record_id
    134               << " stage " << CStage::Label(m_ctx._stage)
    135               ;
    136               
    137     // stage is set by CG4Ctx::setStepOptical from CSteppingAction::setStep
    138     if(m_ctx._stage == CStage::START)
    139     { 
    140         const G4StepPoint* pre = m_ctx._step->GetPreStepPoint() ;
    141         const G4ThreeVector& pos = pre->GetPosition();
    142         m_crec->setOrigin(pos);  
    143         m_crec->clearStp();
    144 
    145         zeroPhoton();       // MUST be invoked prior to setBoundaryStatus, resetting photon history state 
    146 
    147         if(m_dbg) m_dbg->Clear();
    148     }
    149     else if(m_ctx._stage == CStage::REJOIN )
    150     {
    151         if(m_live)
    152         {
    153             decrementSlot();    // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
    154         }
    155         else
    156         {
    157             m_crec->clearStp();
    158            // NB Not-zeroing m_slot for REJOINders, see above note
    159         }
    160     }
    161     else if(m_ctx._stage == CStage::RECOLL )
    162     {
    163         m_decrement_request = 0 ;
    164     }
    165 ...







::

    simon:opticks blyth$ opticks-find SetTrackSecondariesFirst 
    ./cfg4/Cerenkov.cc:void Cerenkov::SetTrackSecondariesFirst(const G4bool state)
    ./cfg4/DsPhysConsOptical.cc:        cerenkov->SetTrackSecondariesFirst(true);
    ./cfg4/DsPhysConsOptical.cc:        cerenkov->SetTrackSecondariesFirst(true);
    ./cfg4/DsPhysConsOptical.cc:    scint->SetTrackSecondariesFirst(true);
    ./cfg4/DsPhysConsOptical.cc:        scint->SetTrackSecondariesFirst(true);
    ./cfg4/OpNovicePhysicsList.cc:        cerenkov->SetTrackSecondariesFirst(true);
    ./cfg4/OpNovicePhysicsList.cc:        cerenkov->SetTrackSecondariesFirst(true);
    ./cfg4/OpNovicePhysicsList.cc:        cerenkov->SetTrackSecondariesFirst(true);
    ./cfg4/OpNovicePhysicsList.cc:    scint->SetTrackSecondariesFirst(true);
    ./cfg4/OpNovicePhysicsList.cc:    scint->SetTrackSecondariesFirst(true);
    ./cfg4/OpNovicePhysicsList.cc:        scint->SetTrackSecondariesFirst(true);
    ./cfg4/OpNovicePhysicsList.cc:  fCerenkovProcess->SetTrackSecondariesFirst(true);
    ./cfg4/OpNovicePhysicsList.cc:  fScintillationProcess->SetTrackSecondariesFirst(true);
    ./cfg4/PhysicsList.cc:  opticalPhysics->SetTrackSecondariesFirst(kCerenkov,true);
    ./cfg4/PhysicsList.cc:  opticalPhysics->SetTrackSecondariesFirst(kScintillation,true);
    ./cfg4/Scintillation.cc:void Scintillation::SetTrackSecondariesFirst(const G4bool state)
    ./cfg4/Cerenkov.hh:        void SetTrackSecondariesFirst(const G4bool state);
    ./cfg4/Scintillation.hh:        void SetTrackSecondariesFirst(const G4bool state);
    ./cfg4/DsG4Cerenkov.h:  void SetTrackSecondariesFirst(const G4bool state);
    ./cfg4/DsG4Cerenkov.h:void DsG4Cerenkov::SetTrackSecondariesFirst(const G4bool state) 
    ./cfg4/DsG4Scintillation.h: void SetTrackSecondariesFirst(const G4bool state);
    ./cfg4/DsG4Scintillation.h:void DsG4Scintillation::SetTrackSecondariesFirst(const G4bool state) 
    simon:opticks blyth$ 



