// g4-

#include "CFG4_PUSH.hh"

#include "G4ProcessManager.hh"
#include "G4RunManager.hh"
#include "G4Event.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4Event.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "DsG4CompositeTrackInfo.h"
#include "DsPhotonTrackInfo.h"




// cg4-
#include "CBoundaryProcess.hh"

#include "CStage.hh"
#include "CGeometry.hh"
#include "CMaterialBridge.hh"
#include "CRecorder.hh"
//#include "Rec.hh"
//#include "State.hh"
#include "Format.hh"
#include "CPropLib.hh"
#include "CStepRec.hh"
#include "CSteppingAction.hh"
#include "CTrack.hh"
#include "CG4.hh"

#include "CFG4_POP.hh"


// optickscore-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

// npy-
#include "PLOG.hh"


#ifdef USE_CUSTOM_BOUNDARY
DsG4OpBoundaryProcessStatus CSteppingAction::GetOpBoundaryProcessStatus()
{
    DsG4OpBoundaryProcessStatus status = Undefined;
#else
G4OpBoundaryProcessStatus CSteppingAction::GetOpBoundaryProcessStatus()
{
    G4OpBoundaryProcessStatus status = Undefined;
#endif
    G4ProcessManager* mgr = G4OpticalPhoton::OpticalPhoton()->GetProcessManager() ;
    if(mgr) 
    {
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcess* opProc = NULL ;  
#else
        G4OpBoundaryProcess* opProc = NULL ;  
#endif
        G4int npmax = mgr->GetPostStepProcessVector()->entries();
        G4ProcessVector* pv = mgr->GetPostStepProcessVector(typeDoIt);
        for (G4int i=0; i<npmax; i++) 
        {
            G4VProcess* proc = (*pv)[i];
#ifdef USE_CUSTOM_BOUNDARY
            opProc = dynamic_cast<DsG4OpBoundaryProcess*>(proc);
#else
            opProc = dynamic_cast<G4OpBoundaryProcess*>(proc);
#endif
            if (opProc) 
            { 
                status = opProc->GetStatus(); 
                break;
            }
        }
    }
    return status ; 
}


/**
CSteppingAction
=================

Canonical instance (m_sa) is ctor resident of CG4 

**/

CSteppingAction::CSteppingAction(CG4* g4, bool dynamic)
   : 
   G4UserSteppingAction(),
   m_g4(g4),
   m_ok(g4->getOpticks()),
   m_dynamic(dynamic),
   m_geometry(g4->getGeometry()),
   m_material_bridge(NULL),
   m_clib(g4->getPropLib()),
   m_recorder(g4->getRecorder()),
   //m_rec(g4->getRec()),
   m_steprec(g4->getStepRec()),
   m_verbosity(m_recorder->getVerbosity()),
   m_event_total(0),
   m_track_total(0),
   m_step_total(0),
   m_event_track_count(0),
   m_track_step_count(0),
   m_steprec_store_count(0),
   m_event(NULL),
   m_track(NULL),
   m_step(NULL),
   m_startEvent(false),
   m_startTrack(false),
   m_event_id(-1),
   m_track_id(-1),
   m_parent_id(-1),
   m_step_id(-1),
   m_primary_id(-1),
   m_track_status(fAlive),
   m_particle(NULL),
   m_optical(false),
   m_pdg_encoding(0)
{ 
}


void CSteppingAction::postinitialize()
{
   // called from CG4::postinitialize
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);
}

CSteppingAction::~CSteppingAction()
{ 
}

const unsigned long long CSteppingAction::SEQHIS_TO_SA = 0x8dull ;     // Torch,SurfaceAbsorb
const unsigned long long CSteppingAction::SEQMAT_MO_PY_BK = 0x5e4ull ; // MineralOil,Pyrex,Bakelite?




void CSteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();
    const G4Event* event = G4RunManager::GetRunManager()->GetCurrentEvent() ;

    int event_id = event->GetEventID() ;

    int track_id = track->GetTrackID() - 1 ;
    int parent_id = track->GetParentID() - 1 ;
    int step_id  = track->GetCurrentStepNumber() - 1 ;

    m_startEvent = m_event_id != event_id ; 
    m_startTrack = m_track_id != track_id || m_startEvent ; 

    if(m_startEvent) setEvent(event, event_id) ; 
    if(m_startTrack) setTrack(track, track_id, parent_id);

    bool done = setStep(step, step_id);

    if(done)
    { 
        track->SetTrackStatus(fStopAndKill);
        // stops tracking when reach truncation as well as absorption
    }
}


void CSteppingAction::setEvent(const G4Event* event, int event_id)
{
    LOG(info) << "CSA (startEvent)"
              << " event_id " << event_id
              << " event_total " <<  m_event_total
              ; 

    m_event = event ; 
    m_event_id = event_id ; 

    m_event_total += 1 ; 
    m_event_track_count = 0 ; 
}

void CSteppingAction::setTrack(const G4Track* track, int track_id, int parent_id)
{
    m_track = track ; 
    m_track_id = track_id ; 
    m_parent_id = parent_id ;
    m_track_status = track->GetTrackStatus(); 

    m_particle = track->GetDefinition();
    m_optical = m_particle == G4OpticalPhoton::OpticalPhotonDefinition() ;
    m_pdg_encoding = m_particle->GetPDGEncoding();

    m_event_track_count += 1 ; 
    m_track_total += 1 ; 

    m_track_step_count = 0 ; 
    m_rejoin_count = 0 ; 

    if(m_optical)
        LOG(trace) << "CSteppingAction::setTrack(optical)"
                  << " track_id " << m_track_id
                  << " parent_id " << m_parent_id
                  ;

}


bool CSteppingAction::setStep(const G4Step* step, int step_id)
{
    bool done = false ; 

    LOG(trace) << "CSteppingAction::setStep" 
               << " step_total " << m_step_total
               ; 

    m_step = step ; 
    m_step_id = step_id ; 

    m_track_step_count += 1 ; 
    m_step_total += 1 ; 

    if(m_optical)
    {
        done = UserSteppingActionOptical(step);
    }
    else
    {
        m_steprec->collectStep(step, step_id);

        if(m_track_status == fStopAndKill)
        {
            done = true ;  
            m_steprec->storeStepsCollected(m_event_id, m_track_id, m_pdg_encoding);
            m_steprec_store_count = m_steprec->getStoreCount(); 
        }
    }

    LOG(debug) << "    (step)"
              << " event_id " << m_event_id
              << " track_id " << m_track_id
              << " track_step_count " << m_track_step_count
              << " step_id " << m_step_id
              << " trackStatus " << CTrack::TrackStatusString(m_track_status)
              ;

   if(m_step_total % 10000 == 0) 
       LOG(debug) << "CSA (totals%10k)"
                 << " track_total " <<  m_track_total
                 << " step_total " <<  m_step_total
                 ;

    return done ;
}



int CSteppingAction::getPrimaryPhotonID()
{
    int primary_id = -2 ; 
    DsG4CompositeTrackInfo* cti = dynamic_cast<DsG4CompositeTrackInfo*>(m_track->GetUserInformation());
    if(cti)
    {
        DsPhotonTrackInfo* pti = dynamic_cast<DsPhotonTrackInfo*>(cti->GetPhotonTrackInfo());
        if(pti)
        {
            primary_id = pti->GetPrimaryPhotonID() ; 
        }
    }
    return primary_id ; 
}


bool CSteppingAction::UserSteppingActionOptical(const G4Step* step)
{
    bool done = false ; 
    int photon_id = m_track_id ;
    int parent_id = m_parent_id ;
    int step_id  = m_step_id ;

    assert( photon_id >= 0 );
    assert( step_id   >= 0 );
    assert( parent_id >= -1 );  // parent_id is -1 for primary photon
  
    int primary_id = getPrimaryPhotonID() ;
    int last_photon_id = m_recorder->getPhotonId();  
    int last_record_id = m_recorder->getRecordId();  

    CStage::CStage_t stage = CStage::UNKNOWN ; 
    if( parent_id == -1 )  // primary photon
    {
        stage = photon_id != last_photon_id  ? CStage::START : CStage::COLLECT ;
    } 
    else if( primary_id >= 0)
    {
        photon_id = primary_id ;  // <-- tacking reem step recording onto primary record 
        stage = m_rejoin_count == 0  ? CStage::REJOIN : CStage::RECOLL ;   
        m_rejoin_count++ ; 
        // rejoin count is zeroed in setTrack, so each remission generation track will result in REJOIN 
    }
     

    if(stage == CStage::START && last_record_id >= 0 && last_record_id < INT_MAX )
    {
        //  backwards looking comparison of prior (potentially rejoined) photon
        //  using record_id for absolute indexing across multiple G4 subevt 
        m_recorder->compare(last_record_id);
    }

 
    m_recorder->setPhotonId(photon_id);   
    m_recorder->setEventId(m_event_id);

    unsigned int record_id = m_recorder->defineRecordId();   //  m_photons_per_g4event*m_event_id + m_photon_id 
    unsigned int record_max = m_recorder->getRecordMax() ;

    bool recording = record_id < record_max ||  m_dynamic ; 

    if(recording)
    {
        m_recorder->setStep(step);
        m_recorder->setStage(stage);
        m_recorder->setStepId(m_step_id);
        m_recorder->setParentId(parent_id);   
        m_recorder->setRecordId(record_id);

        if(stage == CStage::START)
        { 
            m_recorder->startPhoton();  // MUST be invoked from up here,  prior to setBoundaryStatus
            m_recorder->RecordQuadrant(step);
        }
        else if(stage == CStage::REJOIN )
        {
            m_recorder->decrementSlot();    // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
        }

        const G4StepPoint* pre  = step->GetPreStepPoint() ; 
        const G4StepPoint* post = step->GetPostStepPoint() ; 

        const G4Material* preMat  = pre->GetMaterial() ;
        const G4Material* postMat = post->GetMaterial() ;

        unsigned preMaterial = preMat ? m_material_bridge->getMaterialIndex(preMat) + 1 : 0 ;
        unsigned postMaterial = postMat ? m_material_bridge->getMaterialIndex(postMat) + 1 : 0 ;

#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#else
        G4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#endif

        m_recorder->setBoundaryStatus(boundary_status, preMaterial, postMaterial);

        done = m_recorder->RecordStep();   // done=true for *absorption* OR *truncation*

        if(done)
        {
            m_recorder->RecordPhoton(step);
        } 
    }
    return done ; 
}



void CSteppingAction::report(const char* msg)
{
    LOG(info) << msg ;

    std::cout 
           << " event_total " <<  m_event_total << std::endl 
           << " track_total " <<  m_track_total << std::endl 
           << " step_total " <<  m_step_total << std::endl 
           ;

    m_recorder->report(msg);

}

