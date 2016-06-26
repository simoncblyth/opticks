
// g4-
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
#include "CG4.hh"
#include "CRecorder.hh"
#include "Rec.hh"
#include "State.hh"
#include "Format.hh"
#include "CPropLib.hh"
#include "CStepRec.hh"
#include "CSteppingAction.hh"
#include "CTrack.hh"

// optickscore-
#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

// npy-
#include "PLOG.hh"


G4OpBoundaryProcessStatus CSteppingAction::GetOpBoundaryProcessStatus()
{
    G4OpBoundaryProcessStatus status = Undefined;
    G4ProcessManager* mgr = G4OpticalPhoton::OpticalPhoton()->GetProcessManager() ;
    if(mgr) 
    {
        G4OpBoundaryProcess* opProc = NULL ;  
        G4int npmax = mgr->GetPostStepProcessVector()->entries();
        G4ProcessVector* pv = mgr->GetPostStepProcessVector(typeDoIt);
        for (G4int i=0; i<npmax; i++) 
        {
            G4VProcess* proc = (*pv)[i];
            opProc = dynamic_cast<G4OpBoundaryProcess*>(proc);
            if (opProc) 
            { 
                status = opProc->GetStatus(); 
                break;
            }
        }
    }
    return status ; 
}


void CSteppingAction::init()
{
    m_clib = m_g4->getPropLib();
    m_recorder = m_g4->getRecorder();
    m_rec = m_g4->getRec();
    m_steprec = m_g4->getStepRec();

    m_verbosity = m_recorder->getVerbosity(); 
    m_dynamic = m_recorder->isDynamic(); 

    OpticksEvent* evt = m_recorder->getEvent();
    LOG(info) << "CSteppingAction::init " 
              << " evt " << evt->description() 
              << ( m_dynamic ? "DYNAMIC(CPU style)" : "STATIC(GPU style)" )
              ;
}

const unsigned long long CSteppingAction::SEQHIS_TO_SA = 0x8dull ;     // Torch,SurfaceAbsorb
const unsigned long long CSteppingAction::SEQMAT_MO_PY_BK = 0x5e4ull ; // MineralOil,Pyrex,Bakelite?

void CSteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();
    G4TrackStatus status = track->GetTrackStatus();

    int event_id = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() ;
    int track_id = track->GetTrackID() ;
    int parent_id = track->GetParentID() ;
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

       LOG(debug) 
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
    unsigned int eid = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    G4Track* track = step->GetTrack();
    G4int photon_id = track->GetTrackID() - 1;
    G4int step_id  = track->GetCurrentStepNumber() - 1 ;

    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 
    const G4Material* preMat  = pre->GetMaterial() ;
    const G4Material* postMat = post->GetMaterial() ;
    unsigned int preMaterial = preMat ? m_clib->getMaterialIndex(preMat) + 1 : 0 ;
    unsigned int postMaterial = postMat ? m_clib->getMaterialIndex(postMat) + 1 : 0 ;

    bool startPhoton = photon_id != m_recorder->getPhotonId() ; 

    m_recorder->setEventId(eid);
    m_recorder->setStepId(step_id);
    m_recorder->setPhotonId(photon_id);   

    unsigned int record_id = m_recorder->defineRecordId();
    unsigned int record_max = m_recorder->getRecordMax() ;

    if(startPhoton)
         LOG(debug) << "    (opti)"
                   << " photon_id " << photon_id 
                   << " step_id " << step_id 
                   << " record_id " << record_id 
                   << " record_max " << record_max
                   ;

    if(record_id < record_max ||  m_dynamic)
    {
        if(startPhoton)
        { 
            m_rec->Clear();

            m_recorder->startPhoton();  // MUST be invoked from up here,  prior to setBoundaryStatus
            m_recorder->RecordQuadrant(step);
        }

        G4OpBoundaryProcessStatus boundary_status = GetOpBoundaryProcessStatus() ;

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

    //bool debug = rdr_seqmat == SEQMAT_MO_PY_BK && m_verbosity > 0 ;
    bool debug = m_verbosity > 0 ;

    m_rec->setDebug(debug);
    m_rec->sequence();

    unsigned long long rec_seqhis = m_rec->getSeqHis() ;
    unsigned long long rec_seqmat = m_rec->getSeqMat() ;

    bool same_seqhis = rec_seqhis == rdr_seqhis ; 
    bool same_seqmat = rec_seqmat == rdr_seqmat ; 

    assert(same_seqhis);
    assert(same_seqmat);

    if(m_verbosity > 0)
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
             std::cout << "(rec)" << m_clib->MaterialSequence(rec_seqmat) << std::endl ;  
             std::cout << "(rdr)" << m_clib->MaterialSequence(rdr_seqmat) << std::endl ;  
        }
    }
}
