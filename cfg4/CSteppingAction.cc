// g4-

#include "CFG4_PUSH.hh"

#include "G4ProcessManager.hh"
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
#include "Format.hh"
#include "CPropLib.hh"
#include "CStepRec.hh"
#include "CStp.hh"
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
   m_steprec(g4->getStepRec()),
   m_verbosity(m_recorder->getVerbosity()),

   m_event_total(0),
   m_track_total(0),
   m_step_total(0),
   m_event_track_count(0),
   m_steprec_store_count(0),

   m_startEvent(false),
   m_startTrack(false),


   m_event(NULL),
   m_event_id(-1),

   m_track_step_count(0),

   m_track(NULL),
   m_track_id(-1),
   m_optical(false),
   m_pdg_encoding(0),

   m_photon_id(0),
   m_reemtrack(false),
   m_rejoin_count(0),
   m_primarystep_count(0),

   m_step(NULL),
   m_step_id(-1)
{ 
}

int CSteppingAction::getStepId()
{
    return m_step_id ; 
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


void CSteppingAction::setEvent(const G4Event* event, int event_id)
{
    LOG(debug) << "CSA (setEvent)"
              << " event_id " << event_id
              << " event_total " <<  m_event_total
              ; 

    m_event = event ; 
    m_event_id = event_id ; 

    m_event_total += 1 ; 
    m_event_track_count = 0 ; 
}

void CSteppingAction::setTrack(const G4Track* track, int track_id, bool optical, int pdg_encoding )
{
    m_track = track ; 
    m_track_id = track_id ; 
    m_optical = optical ; 
    m_pdg_encoding = pdg_encoding ; 

    ///////////////////////////////////
    m_track_step_count = 0 ; 
    m_event_track_count += 1 ; 
    m_track_total += 1 ; 
}

void CSteppingAction::setPhotonId(int photon_id, bool reemtrack)
{
    assert( photon_id >= 0 );
    m_photon_id = photon_id ; 
    m_reemtrack = reemtrack ; 
    m_rejoin_count = 0 ; 
    m_primarystep_count = 0 ; 

    LOG(debug) << "CSteppingAction::setPhotonId"
              << " event_id " << m_event_id 
              << " track_id " << m_track_id 
              << " photon_id " << photon_id 
              << " reemtrack " << reemtrack
              ; 
}


void CSteppingAction::setRecordId(int record_id, bool dbg, bool other)
{
    m_record_id = record_id ; 
    m_debug = dbg  ; 
    m_other = other  ; 
}



/// above methods are invoked from on high by CTrackingAction prior to getting any steps

void CSteppingAction::UserSteppingAction(const G4Step* step)
{
    int step_id = CTrack::StepId(m_track);
    bool done = setStep(step, step_id);

    if(done)
    { 
        G4Track* track = step->GetTrack();    // m_track is const qualified
        track->SetTrackStatus(fStopAndKill);
        // stops tracking when reach truncation as well as absorption
    }
}

const G4ThreeVector& CSteppingAction::getStepOrigin()
{
    return m_step_origin ;
}


bool CSteppingAction::setStep(const G4Step* step, int step_id)
{
    bool done = false ; 

    m_step = step ; 
    m_step_id = step_id ; 

    if(m_step_id == 0)
    {
        const G4StepPoint* pre = m_step->GetPreStepPoint() ;
        m_step_origin = pre->GetPosition();
    }


    m_track_step_count += 1 ; 
    m_step_total += 1 ; 

    G4TrackStatus track_status = m_track->GetTrackStatus(); 

    LOG(trace) << "CSteppingAction::setStep" 
              << " step_total " << m_step_total
              << " event_id " << m_event_id
              << " track_id " << m_track_id
              << " track_step_count " << m_track_step_count
              << " step_id " << m_step_id
              << " trackStatus " << CTrack::TrackStatusString(track_status)
              ;

    if(m_optical)
    {
        done = collectPhotonStep();
    }
    else
    {
        m_steprec->collectStep(step, step_id);
    
        if(track_status == fStopAndKill)
        {
            done = true ;  
            m_steprec->storeStepsCollected(m_event_id, m_track_id, m_pdg_encoding);
            m_steprec_store_count = m_steprec->getStoreCount(); 
        }
    }

   if(m_step_total % 10000 == 0) 
       LOG(debug) << "CSA (totals%10k)"
                 << " track_total " <<  m_track_total
                 << " step_total " <<  m_step_total
                 ;

    return done ;
}


bool CSteppingAction::collectPhotonStep()
{
    bool done = false ; 


    CStage::CStage_t stage = CStage::UNKNOWN ; 

    if( !m_reemtrack )     // primary photon, ie not downstream from reemission 
    {
        stage = m_primarystep_count == 0  ? CStage::START : CStage::COLLECT ;
        m_primarystep_count++ ; 
    } 
    else 
    {
        stage = m_rejoin_count == 0  ? CStage::REJOIN : CStage::RECOLL ;   
        m_rejoin_count++ ; 
        // rejoin count is zeroed in setPhotonId, so each remission generation trk will result in REJOIN 
    }


    // TODO: avoid need for these
    m_recorder->setPhotonId(m_photon_id);   
    m_recorder->setEventId(m_event_id);

    int record_max = m_recorder->getRecordMax() ;
    bool recording = m_record_id < record_max ||  m_dynamic ; 

    if(recording)
    {
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#else
        G4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#endif
        done = m_recorder->Record(m_step, m_step_id, m_record_id, m_debug, m_other, boundary_status, stage);

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

