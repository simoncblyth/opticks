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



cfg4 reemission
-----------------

::

    epsilon:cfg4 blyth$ grep reemission *.cc
    CG4Ctx.cc:     // retaining original photon_id from prior to reemission effects the continuation
    CG4Ctx.cc:    if( !_reemtrack )     // primary photon, ie not downstream from reemission 
    CPropLib.cc:    bool is_scintillator = _ggmat->hasNonZeroProperty("reemission_prob") ;
    CPropLib.cc:    assert(scintillator && "non-zero reemission prob materials should has an associated raw scintillator");
    CPropLib.cc:        const char* key =  pmap->getPropertyNameByIndex(i); // refractive_index absorption_length scattering_length reemission_prob
    CRecorder.cc:* see notes/issues/reemission_review.rst
    CRecorder.cc:via the record_id (which survives reemission) the info is written 
    CWriter.cc:*hard_truncate* does happen for top slot without reemission rejoinders
    CWriter.cc:for reemission have to rely on downstream overwrites
    DsG4Scintillation.cc:        G4double p_reemission= Reemission_Prob->Value(aTrack.GetKineticEnergy());
    DsG4Scintillation.cc:        G4double p_reemission= Reemission_Prob->GetProperty(aTrack.GetKineticEnergy());
    DsG4Scintillation.cc:        if (G4UniformRand() >= p_reemission) return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    DsG4Scintillation.cc:                // reemission, the sample method need modification
    DsG4Scintillation.cc:                trackinf->SetPrimaryPhotonID( primary_id ); // SCB for reemission continuation recording 
    DsG4Scintillation.cc://  For Opticks style reemission continuation recording need to 
    DsG4Scintillation.cc://  trace reemission lineage back to first photon.
    DsG4Scintillation.cc://  The below attempts at each reemission generation to pass 
    DsG4Scintillation.cc://  along this primary index unchanged, so reemission photons 
    DsPhysConsOptical.cc:      m_doReemission(true),               // "ScintDoReemission"        "Do reemission in scintilator."
    OpNovicePhysicsList.cc:    m_doReemission(true),               // "ScintDoReemission"        "Do reemission in scintilator."
    epsilon:cfg4 blyth$ 


* TODO: compare this old DsG4Scintillation with the current JUNO one 



how lineage was passed thru RE-emission generations
------------------------------------------------------

* TODO: pass the primary record id in cfg4/CTrackInfo.hh


::

     201 G4VParticleChange*
     202 DsG4Scintillation::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     203 
     204 // This routine is called for each tracking step of a charged particle
     205 // in a scintillator. A Poisson/Gauss-distributed number of photons is 
     206 // generated according to the scintillation yield formula, distributed 
     207 // evenly along the track segment and uniformly into 4pi.
     208 
     ...
     826             DsG4CompositeTrackInfo* comp=new DsG4CompositeTrackInfo();
     827             DsPhotonTrackInfo* trackinf=new DsPhotonTrackInfo();
     828             if ( flagReemission )
     829             {
     830                 if ( reemittedTI ) *trackinf = *reemittedTI;
     831                 trackinf->SetReemitted();
     832                 int primary_id = getReemissionPrimaryPhotonID(aTrack, aSecondaryTime);
     833                 trackinf->SetPrimaryPhotonID( primary_id ); // SCB for reemission continuation recording 
     834             }
     835             else if ( fApplyPreQE ) {
     836                 trackinf->SetMode(DsPhotonTrackInfo::kQEPreScale);
     837                 trackinf->SetQE(fPreQE);
     838             }
     839             comp->SetPhotonTrackInfo(trackinf);
     840             aSecondaryTrack->SetUserInformation(comp);
     841    
     842             aSecondaryTrack->SetParentID(aTrack.GetTrackID()) ;
     843             aSecondaryTrack->SetWeight( weight );
     844             aSecondaryTrack->SetTouchableHandle(aStep.GetPreStepPoint()->GetTouchableHandle());
     845             // aSecondaryTrack->SetTouchableHandle((G4VTouchable*)0);//this is wrong
     846 



::

     873 int DsG4Scintillation::getReemissionPrimaryPhotonID(const G4Track& aTrack, G4double /*aSecondaryTime*/)
     874 {
     875 // SCB
     876 //  For Opticks style reemission continuation recording need to 
     877 //  trace reemission lineage back to first photon.
     878 //
     879 //  Initially tried  hijacking the "secondary-tracking" ParentID 
     880 //  but G4 stomped on that approach, so using trackinfo to hold PrimaryPhotonID 
     881 //
     882 //  The below attempts at each reemission generation to pass 
     883 //  along this primary index unchanged, so reemission photons 
     884 //  stay associated thru the  generations back to the primary photon id.
     885 //
     886 //  This makes an assumption that multi-reemits are handled 
     887 //  in subsequent optical calls to DsG4Scintillation::PostStepDoItProc
     888 //  

     MAY 2021 COMMENT : THINK THAT THERE IS NO SUCH ASSUMPTION HERE, BUT THERE IS ELSEWHERE 
     FOR RECORDING OF RE-JOIN CONTINUATIONS 

     889 
     890     int track_id = aTrack.GetTrackID() - 1 ;
     891     int parent_id = aTrack.GetParentID() - 1 ;
     892     int primary_id = -1 ;
     893 
     894    // TODO: replace m_lineage with simply  m_primary_id ??? 
     895    //       perhaps this should be using record_id for absolute indexing ??
     896 
     897     if(parent_id == -1)  // primary photon 
     898     {
     899         m_lineage.clear() ;
     900         primary_id  = track_id ;
     901         m_lineage.push_back(primary_id);
     902     }
     903     else
     904     {
     905         m_lineage.push_back(parent_id) ;
     906         primary_id = m_lineage.front() ;
     907     }
     908 
     909 
     910 /* 
     911     LOG(info) << " DsG4Scintillation::getReemissionPrimaryPhotonID" 
     912               << " psdi_index " << m_psdi_index
     913               << " secondaryTime(ns) " << aSecondaryTime/ns 
     914               << " track_id " << track_id
     915               << " parent_id " << parent_id
     916               << " primary_id " << primary_id
     917               << " lineage " << m_lineage.size()
     918               ;
     919 
     920     std::cout << " lineage (" ;
     921     for(std::vector<int>::const_iterator it=m_lineage.begin() ; it != m_lineage.end() ; it++) std::cout << *it << " " ; 
     922     std::cout << ")" << std::endl  ;  
     923 */
     924 
     925 
     926     return primary_id ;
     927 }




::

    epsilon:cfg4 blyth$ grep SetPrimaryPhotonID *.*
    DsG4Scintillation.cc:                trackinf->SetPrimaryPhotonID( primary_id ); // SCB for reemission continuation recording 
    DsPhotonTrackInfo.h:    void  SetPrimaryPhotonID(int ppi){ fPrimaryPhotonID = ppi ; ; }
    epsilon:cfg4 blyth$ 



::

     22 DsPhotonTrackInfo::DsPhotonTrackInfo( QEMode mode, double qe )
     23     :
     24     fMode(mode),
     25     fQE(qe),
     26     fReemitted(false),
     27     fPrimaryPhotonID(-1)
     28 {
     29 }
     30 

     27 class CFG4_API DsPhotonTrackInfo : public G4VUserTrackInformation
     28 {
     29 public:
     30     enum QEMode
     31     {
     32             kQENone, 
     33             kQEPreScale, 
     34             kQEWater 
     35     };
     36 
     37     DsPhotonTrackInfo(QEMode mode=DsPhotonTrackInfo::kQENone, double qe=1.) ;
     38 
     39 
     40     QEMode GetMode() { return fMode; }
     41     void   SetMode(QEMode m) { fMode=m; }
     42 
     43     double GetQE() { return fQE; }
     44     void   SetQE(double qe) { fQE=qe; }
     45 
     46     bool GetReemitted() { return fReemitted; }
     47     void SetReemitted( bool re=true ) { fReemitted=re; }
     48 
     49     void  SetPrimaryPhotonID(int ppi){ fPrimaryPhotonID = ppi ; ; }
     50     int   GetPrimaryPhotonID(){ return fPrimaryPhotonID ; } 
     51     
     52     void Print() const {};
     53 private:
     54     QEMode fMode;
     55     double fQE;
     56     bool   fReemitted;
     57     int    fPrimaryPhotonID  ;
     58 };


::

    epsilon:cfg4 blyth$ grep DsPhotonTrackInfo *.*
    CMakeLists.txt:    DsPhotonTrackInfo.cc
    CMakeLists.txt:    DsPhotonTrackInfo.h
    CTrack.cc:#include "DsPhotonTrackInfo.h"
    CTrack.cc:        DsPhotonTrackInfo* pti = dynamic_cast<DsPhotonTrackInfo*>(cti->GetPhotonTrackInfo());
    DsG4Cerenkov.cc:#include "DsPhotonTrackInfo.h"
    DsG4Cerenkov.cc:	  DsPhotonTrackInfo* trackinf=new DsPhotonTrackInfo();
    DsG4Cerenkov.cc:	    trackinf->SetMode(DsPhotonTrackInfo::kQEWater);
    DsG4Cerenkov.cc:	    trackinf->SetMode(DsPhotonTrackInfo::kQEPreScale);
    DsG4Scintillation.cc:#include "DsPhotonTrackInfo.h"
    DsG4Scintillation.cc:    DsPhotonTrackInfo* reemittedTI=0;
    DsG4Scintillation.cc:        reemittedTI = composite?dynamic_cast<DsPhotonTrackInfo*>( composite->GetPhotonTrackInfo() ):0;
    DsG4Scintillation.cc:            DsPhotonTrackInfo* trackinf=new DsPhotonTrackInfo();
    DsG4Scintillation.cc:                trackinf->SetMode(DsPhotonTrackInfo::kQEPreScale);
    DsPhotonTrackInfo.cc:#include "DsPhotonTrackInfo.h"
    DsPhotonTrackInfo.cc:DsPhotonTrackInfo::DsPhotonTrackInfo( QEMode mode, double qe )
    DsPhotonTrackInfo.h:class CFG4_API DsPhotonTrackInfo : public G4VUserTrackInformation
    DsPhotonTrackInfo.h:    DsPhotonTrackInfo(QEMode mode=DsPhotonTrackInfo::kQENone, double qe=1.) ;
    epsilon:cfg4 blyth$ 




how the PrimaryPhotonID is used to effect RE-JOINING 
----------------------------------------------------------- 

Critical code for this::


    403 void CG4Ctx::setTrackOptical()
    404 {
    405     LOG(debug) << "CTrackingAction::setTrack setting UseGivenVelocity for optical " ;
    406     
    407     _track->UseGivenVelocity(true);
    408     
    409     // NB without this BoundaryProcess proposed velocity to get correct GROUPVEL for material after refraction 
    410     //    are trumpled by G4Track::CalculateVelocity 
    411     
    412     // _primary_id = CTrack::PrimaryPhotonID(_track) ;    // layed down in trackinfo by custom Scintillation process
    413     // _photon_id = _primary_id >= 0 ? _primary_id : _track_id ; 
    414     
    415     
    416     // dynamic_cast gives NULL when using the wrong type for the pointer
    417     CTrackInfo* tkui = dynamic_cast<CTrackInfo*>(_track->GetUserInformation());   // NEW ADDITION : NEEDS INTEGRATING 
    418     _primary_id = tkui ? tkui->opticks_photon_id() : -1 ; 
    419     char tkui_gentype = tkui ? tkui->opticks_gentype() : '?' ;
    420     
    421     assert( _primary_id >= 0 && tkui_gentype != '?' );   // require all optical tracks to have been annotated with CTrackInfo 
    422     _photon_id = _primary_id  ; 
    423     
    424      
    425      // HUH: surely passing down the primary will mean _primary_id always >= 0 
    426      // what eactly was the old CTrack::PrimaryPhotonID ?
    427      // 
    428     _reemtrack = _primary_id >= 0 ? true : false ; // <-- critical input to _stage set by subsequent CG4Ctx::setStepOptical 
    429     _photon_count += 1 ;
    430 
    431 
    432      // retaining original photon_id from prior to reemission effects the continuation
    433     _record_id = _photons_per_g4event*_event_id + _photon_id ;
    434     _record_fraction = double(_record_id)/double(_record_max) ;
    435 
    436     LOG(LEVEL)
    437         << " _record_id " << _record_id
    438         << " _primary_id " << _primary_id
    439         << " tkui_gentype " << tkui_gentype
    440         ;
    441     setGen(tkui_gentype);
    442 
    443 





Below is currently as mix of old and new ways of doing things
-----------------------------------------------------------------


::

    390 void CG4Ctx::setTrackOptical()
    391 {
    392     LOG(debug) << "CTrackingAction::setTrack setting UseGivenVelocity for optical " ;
    393 
    394     _track->UseGivenVelocity(true);
    395 
    396     // NB without this BoundaryProcess proposed velocity to get correct GROUPVEL for material after refraction 
    397     //    are trumpled by G4Track::CalculateVelocity 
    398 
    399     _primary_id = CTrack::PrimaryPhotonID(_track) ;    // layed down in trackinfo by custom Scintillation process
    400     _photon_id = _primary_id >= 0 ? _primary_id : _track_id ;
    401     _reemtrack = _primary_id >= 0 ? true        : false ;
    402     _photon_count += 1 ;
    403 
    404      // retaining original photon_id from prior to reemission effects the continuation
    405     _record_id = _photons_per_g4event*_event_id + _photon_id ;
    406     _record_fraction = double(_record_id)/double(_record_max) ;
    407 
    408     // dynamic_cast gives NULL when using the wrong type for the pointer
    409     CTrackInfo* tkui = dynamic_cast<CTrackInfo*>(_track->GetUserInformation());
    410     _tk_record_id = tkui ? tkui->record_id() : -1 ;
    411     char tk_gentype = tkui ? tkui->gentype() : '?' ;
    412 
    413     LOG(LEVEL)
    414         << " _record_id " << _record_id
    415         << " _tk_record_id " << _tk_record_id
    416         << " tk_gentype " << tk_gentype
    417         ;
    418     setGen(tk_gentype);



Hmm the below should be double templated to make plain the tracking class type assumptions::

    092 int CTrack::PrimaryPhotonID(const G4Track* track)
     93 {
     94     int primary_id = -2 ;
     95     DsG4CompositeTrackInfo* cti = dynamic_cast<DsG4CompositeTrackInfo*>(track->GetUserInformation());
     96     if(cti)
     97     {
     98         DsPhotonTrackInfo* pti = dynamic_cast<DsPhotonTrackInfo*>(cti->GetPhotonTrackInfo());
     99         if(pti)
    100         {
    101             primary_id = pti->GetPrimaryPhotonID() ;
    102         }
    103     }
    104     return primary_id ;
    105 }




opticks_photon_id
---------------------







