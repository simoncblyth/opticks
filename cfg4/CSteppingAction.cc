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
#include "CG4Ctx.hh"
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
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks()),
   m_dbgrec(m_ok->isDbgRec()),
   m_dynamic(dynamic),
   m_geometry(g4->getGeometry()),
   m_material_bridge(NULL),
   m_mlib(g4->getMaterialLib()),
   m_recorder(g4->getRecorder()),
   m_steprec(g4->getStepRec()),
   m_verbosity(m_recorder->getVerbosity()),

   m_step_total(0),
   m_steprec_store_count(0),

   m_startEvent(false),
   m_startTrack(false),

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

/// above methods are invoked from on high by CTrackingAction prior to getting any steps

void CSteppingAction::UserSteppingAction(const G4Step* step)
{
    int step_id = CTrack::StepId(m_ctx._track);
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

void CSteppingAction::setPhotonId()
{
    assert( m_ctx._photon_id >= 0 );

    m_rejoin_count = 0 ; 
    m_primarystep_count = 0 ; 

    LOG(debug) << "CSteppingAction::setPhotonId"
              << " event_id " << m_ctx._event_id 
              << " track_id " << m_ctx._track_id 
              << " photon_id " << m_ctx._photon_id 
              << " reemtrack " << m_ctx._reemtrack
              ; 
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

    m_ctx._track_step_count += 1 ; 

    m_step_total += 1 ; 

    G4TrackStatus track_status = m_ctx._track->GetTrackStatus(); 

    LOG(trace) << "CSteppingAction::setStep" 
              << " step_total " << m_step_total
              << " event_id " << m_ctx._event_id
              << " track_id " << m_ctx._track_id
              << " track_step_count " << m_ctx._track_step_count
              << " step_id " << m_step_id
              << " trackStatus " << CTrack::TrackStatusString(track_status)
              ;

    if(m_ctx._optical)
    {
        done = collectPhotonStep();
    }
    else
    {
        m_steprec->collectStep(step, step_id);
    
        if(track_status == fStopAndKill)
        {
            done = true ;  
            m_steprec->storeStepsCollected(m_ctx._event_id, m_ctx._track_id, m_ctx._pdg_encoding);
            m_steprec_store_count = m_steprec->getStoreCount(); 
        }
    }

   if(m_step_total % 10000 == 0) 
       LOG(debug) << "CSA (totals%10k)"
                 << " track_total " <<  m_ctx._track_total
                 << " step_total " <<  m_step_total
                 ;

    return done ;
}


bool CSteppingAction::collectPhotonStep()
{
    bool done = false ; 


    CStage::CStage_t stage = CStage::UNKNOWN ; 

    if( !m_ctx._reemtrack )     // primary photon, ie not downstream from reemission 
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


    int record_max = m_recorder->getRecordMax() ;
    bool recording = m_ctx._record_id < record_max ||  m_dynamic ;  // record_max is a photon level fit-in-buffer thing
   


    if(recording)
    {
#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#else
        G4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#endif
        done = m_recorder->Record(m_step, m_step_id, boundary_status, stage);

    }
    return done ; 
}

void CSteppingAction::report(const char* msg)
{
    LOG(info) << msg ;
    std::cout 
           << " event_total " <<  m_ctx._event_total << std::endl 
           << " track_total " <<  m_ctx._track_total << std::endl 
           << " step_total " <<  m_step_total << std::endl 
           ;
    m_recorder->report(msg);
}

