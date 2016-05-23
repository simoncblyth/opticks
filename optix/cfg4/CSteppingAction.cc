
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
#include "Recorder.hh"
#include "Rec.hh"
#include "State.hh"
#include "Format.hh"
#include "CPropLib.hh"
#include "CStepRec.hh"
#include "CSteppingAction.hh"

// optickscore-
#include "Opticks.hh"

// npy-
#include "NLog.hpp"


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

    NumpyEvt* evt = m_recorder->getEvt();
    LOG(info) << "CSteppingAction::init " 
              << " evt " << evt->description() 
              ;
}



const unsigned long long CSteppingAction::SEQHIS_TO_SA = 0x8dull ;     // Torch,SurfaceAbsorb
const unsigned long long CSteppingAction::SEQMAT_MO_PY_BK = 0x5e4ull ; // MineralOil,Pyrex,Bakelite?


void CSteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();

    int event_id = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID() ;
    int track_id = track->GetTrackID() ;
    //int parent_id = track->GetParentID() ;
    int step_id  = track->GetCurrentStepNumber() - 1 ;

    bool startEvent = m_event_id != event_id ; 
    bool startTrack = m_track_id != track_id || startEvent ; 

    const G4ParticleDefinition* type = track->GetDefinition();
    const G4ParticleDefinition* type2 = track->GetDynamicParticle()->GetParticleDefinition(); 
    assert( type == type2 );

    G4String particleName = type->GetParticleName();
    G4int pdgCode = type->GetPDGEncoding();

    setEventId(event_id);     
    setTrackId(track_id);     
    //setParentId(parent_id);
    //setPDGCode(pdgCode);

    if( type == G4OpticalPhoton::OpticalPhotonDefinition())
    {
        LOG(info) << "CSteppingAction::UserSteppingAction skip optical " ; 
        //UserSteppingActionOptical(step);
    }
    else
    {
        if(startTrack) 
        {
            m_steprec->store(event_id, track_id, pdgCode);
        }
        m_steprec->record(step, step_id);
        // hmm this will miss the last track
    }
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

    /*
    if(startPhoton)
         LOG(info) << "CSteppingAction::UserSteppingAction"
                   << " photon_id " << photon_id 
                   << " step_id " << step_id 
                   << " record_id " << record_id 
                   << " record_max " << record_max
                   ;
    */

    if(record_id < record_max)
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

        m_rec->add(new State(step, boundary_status, preMaterial, postMaterial));
        if(done)
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

            m_recorder->RecordPhoton(step);

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
                     std::cout << "(rec)" << Opticks::FlagSequence(rec_seqhis) << std::endl ;  
                     std::cout << "(rdr)" << Opticks::FlagSequence(rdr_seqhis) << std::endl ;  
                }
                if(!same_seqmat)
                { 
                     std::cout << "(rec)" << m_clib->MaterialSequence(rec_seqmat) << std::endl ;  
                     std::cout << "(rdr)" << m_clib->MaterialSequence(rdr_seqmat) << std::endl ;  
                }


            }

            track->SetTrackStatus(fStopAndKill);
        } 
    }

}





