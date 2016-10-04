#include "CFG4_BODY.hh"
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

// cg4-
#include "CBoundaryProcess.hh"

#include "CGeometry.hh"
#include "CMaterialBridge.hh"
#include "CRecorder.hh"
#include "Rec.hh"
#include "State.hh"
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
   m_dbgseqhis(m_ok->getDbgSeqhis()),
   m_dbgseqmat(m_ok->getDbgSeqmat()),
   m_dynamic(dynamic),
   m_geometry(g4->getGeometry()),
   m_material_bridge(NULL),
   m_clib(g4->getPropLib()),
   m_recorder(g4->getRecorder()),
   m_rec(g4->getRec()),
   m_steprec(g4->getStepRec()),
   m_verbosity(m_recorder->getVerbosity()),
   m_event_id(-1),
   m_track_id(-1),
   m_event_total(0),
   m_track_total(0),
   m_step_total(0),
   m_event_track_count(0),
   m_track_step_count(0),
   m_steprec_store_count(0)
{ 
   init();
}

void CSteppingAction::init()
{
    LOG(fatal) << "CSteppingAction::init " 
              << ( m_dynamic ? "DYNAMIC(CPU style)" : "STATIC(GPU style)" )
              ;
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


void CSteppingAction::setTrackId(unsigned int track_id)
{
    m_track_id = track_id ; 
}
void CSteppingAction::setEventId(unsigned int event_id)
{
    m_event_id = event_id ; 
}




const unsigned long long CSteppingAction::SEQHIS_TO_SA = 0x8dull ;     // Torch,SurfaceAbsorb
const unsigned long long CSteppingAction::SEQMAT_MO_PY_BK = 0x5e4ull ; // MineralOil,Pyrex,Bakelite?

void CSteppingAction::UserSteppingAction(const G4Step* step)
{

    LOG(trace) << "CSteppingAction::UserSteppingAction" 
               << " step_total " << m_step_total
               ; 

    G4Track* track = step->GetTrack();
    G4TrackStatus status = track->GetTrackStatus();

    int event_id = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() ;
    int track_id = track->GetTrackID() ;
    //int parent_id = track->GetParentID() ;
    int step_id  = track->GetCurrentStepNumber() - 1 ;

    bool startEvent = m_event_id != event_id ; 
    bool startTrack = m_track_id != track_id || startEvent ; 

    if(startEvent)
    {
       LOG(info) << "CSA (startEvent)"
                 << " event_id " << event_id
                 << " event_total " <<  m_event_total
                 ; 

       m_event_total += 1 ; 
       m_event_track_count = 0 ; 
    }

    static G4ParticleDefinition* type ; 
    static bool optical ; 
    static G4String particle_name ; 
    static G4int pdg_encoding ;
    
    if(startTrack)
    {
       type = track->GetDefinition();
       optical = type == G4OpticalPhoton::OpticalPhotonDefinition() ;
       particle_name = type->GetParticleName();
       pdg_encoding = type->GetPDGEncoding();

/*
       LOG(trace) 
                 << "CSA (trak)"
                 << " event_id " << event_id
                 << " track_id " << track_id
                 << " parent_id " << parent_id
                 << " event_track_count " <<  m_event_track_count
                 << " pdg_encoding " << pdg_encoding
                 << " optical " << optical 
                 << " particle_name " << particle_name 
                 << " steprec_store_count " << m_steprec_store_count
                 ; 
*/
       m_event_track_count += 1 ; 
       m_track_step_count = 0 ; 
       m_track_total += 1 ; 
    }

    m_track_step_count += 1 ; 
    m_step_total += 1 ; 


    setEventId(event_id);     
    setTrackId(track_id);     

    if(optical)
    {
        UserSteppingActionOptical(step);
    }
    else
    {
        m_steprec->collectStep(step, step_id);
        if(status == fStopAndKill)
        {
            m_steprec->storeStepsCollected(event_id, track_id, pdg_encoding);
            m_steprec_store_count = m_steprec->getStoreCount(); 
        }
    }

    LOG(debug) << "    (step)"
              << " event_id " << event_id
              << " track_id " << track_id
              << " track_step_count " << m_track_step_count
              << " step_id " << step_id
              << " trackStatus " << CTrack::TrackStatusString(status)
              ;

   if(m_step_total % 10000 == 0) 
       LOG(debug) << "CSA (totals%10k)"
                 << " track_total " <<  m_track_total
                 << " step_total " <<  m_step_total
                 ;
}



void CSteppingAction::UserSteppingActionOptical(const G4Step* step)
{
    //LOG(trace) << "CSA::UserSteppingActionOptical" ; 

    unsigned int eid = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    G4Track* track = step->GetTrack();

    int photon_id = track->GetTrackID() - 1;
    int parent_id = track->GetParentID() - 1 ;
    int step_id  = track->GetCurrentStepNumber() - 1 ;

    assert( photon_id >= 0 );
    assert( step_id   >= 0 );
    assert( parent_id >= -1 );  // parent_id is -1 for non-secondary tracks 

    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    const G4Material* preMat  = pre->GetMaterial() ;
    const G4Material* postMat = post->GetMaterial() ;

    unsigned preMaterial = preMat ? m_material_bridge->getMaterialIndex(preMat) + 1 : 0 ;
    unsigned postMaterial = postMat ? m_material_bridge->getMaterialIndex(postMat) + 1 : 0 ;
   

    bool startPhoton = photon_id != m_recorder->getPhotonId() ; 
    bool continuePhoton = parent_id >= 0  ;

    m_recorder->setEventId(eid);
    m_recorder->setStepId(step_id);
    m_recorder->setPhotonId(photon_id);   
    m_recorder->setParentId(parent_id);   

    unsigned int record_id = m_recorder->defineRecordId();   //  m_photons_per_g4event*m_event_id + m_photon_id 
    unsigned int record_max = m_recorder->getRecordMax() ;

    bool stepRecord = record_id < record_max ||  m_dynamic ; 

   // slot continuation for reemission means need to find the prior id 

    if(startPhoton || continuePhoton || !stepRecord)
         LOG(info) 
                   << ( startPhoton     ? "S" : "-" )
                   << ( continuePhoton  ? "C" : "-" )
                   << ( stepRecord      ? "R" : "-" )
                   << " photon_id " << std::setw(7) << photon_id 
                   << " parent_id " << std::setw(7) << parent_id
                   << " step_id " << std::setw(4) << step_id 
                   << " record_id " << std::setw(7) << record_id 
                   << " record_max " << std::setw(7) << record_max
                  << ( m_dynamic ? " DYNAMIC " : " STATIC " )
                   ;


    if(stepRecord)
    {
        if(startPhoton)
        { 
            m_rec->Clear();

            m_recorder->startPhoton();  // MUST be invoked from up here,  prior to setBoundaryStatus
            m_recorder->RecordQuadrant(step);
        }

#ifdef USE_CUSTOM_BOUNDARY
        DsG4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#else
        G4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;
#endif

        m_recorder->setRecordId(record_id);
        m_recorder->setBoundaryStatus(boundary_status, preMaterial, postMaterial);

        bool done = m_recorder->RecordStep(step);  

        //   *absorption* of photon
        //   *truncation* when get to maximum records to store 

        m_rec->add(new State(step, boundary_status, preMaterial, postMaterial));

        if(done)
        {
            compareRecords();

            m_recorder->RecordPhoton(step);

            track->SetTrackStatus(fStopAndKill);

            // stops tracking when reach truncation 
            // for absorption, this should already be set  ? TODO:check
        } 
    }


}

void CSteppingAction::compareRecords()
{
    unsigned long long rdr_seqhis = m_recorder->getSeqHis() ;
    unsigned long long rdr_seqmat = m_recorder->getSeqMat() ;

    bool debug_seqhis = m_dbgseqhis == rdr_seqhis ; 
    bool debug_seqmat = m_dbgseqmat == rdr_seqmat ; 

    //bool debug = rdr_seqmat == SEQMAT_MO_PY_BK && m_verbosity > 0 ;
    bool debug = m_verbosity > 0 || debug_seqhis || debug_seqmat ;

    m_rec->setDebug(debug);
    m_rec->sequence();

    unsigned long long rec_seqhis = m_rec->getSeqHis() ;
    unsigned long long rec_seqmat = m_rec->getSeqMat() ;

    bool same_seqhis = rec_seqhis == rdr_seqhis ; 
    bool same_seqmat = rec_seqmat == rdr_seqmat ; 

    assert(same_seqhis);
    assert(same_seqmat);

    if(m_verbosity > 0 || debug )
    {
        if(!same_seqmat || !same_seqhis || debug )
        {
            std::cout << std::endl << std::endl << "----CSteppingAction----" << std::endl  ; 
            m_recorder->Dump("CSteppingAction::UserSteppingAction DONE");
            m_rec->Dump("CSteppingAction::UserSteppingAction (Rec)DONE");
        }

        if(!same_seqhis)
        { 
             std::cout << "(rec)" << OpticksFlags::FlagSequence(rec_seqhis) << std::endl ;  
             std::cout << "(rdr)" << OpticksFlags::FlagSequence(rdr_seqhis) << std::endl ;  
        }
        if(!same_seqmat)
        { 
             std::cout << "(rec)" << m_material_bridge->MaterialSequence(rec_seqmat) << std::endl ;  
             std::cout << "(rdr)" << m_material_bridge->MaterialSequence(rdr_seqmat) << std::endl ;  
        }
    }
}
