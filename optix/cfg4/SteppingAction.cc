#include "SteppingAction.hh"

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

#include "Recorder.hh"
#include "Format.hh"

#include "NLog.hpp"


G4OpBoundaryProcessStatus SteppingAction::GetOpBoundaryProcessStatus()
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


void SteppingAction::init()
{
    NumpyEvt* evt = m_recorder->getEvt();

    m_bounce_max = evt->getBounceMax();

    LOG(info) << "SteppingAction::init " 
              << " evt " << evt->description() 
              << " m_bounce_max  " << m_bounce_max 
              ;
}



void SteppingAction::UserSteppingAction(const G4Step* step)
{
    unsigned int eid = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    G4Track* track = step->GetTrack();
    G4int photon_id = track->GetTrackID() - 1;
    G4int step_id  = track->GetCurrentStepNumber() - 1 ;

    bool startPhoton = photon_id != m_recorder->getPhotonId() ; 

    m_recorder->setEventId(eid);
    m_recorder->setStepId(step_id);
    m_recorder->setPhotonId(photon_id);   

    unsigned int record_id = m_recorder->defineRecordId();
    unsigned int record_max = m_recorder->getRecordMax() ;

    /*
    if(startPhoton)
         LOG(info) << "SteppingAction::UserSteppingAction"
                   << " photon_id " << photon_id 
                   << " step_id " << step_id 
                   << " record_id " << record_id 
                   << " record_max " << record_max
                   ;
    */

    if(record_id < record_max)
    {
        m_recorder->setRecordId(record_id);
        if(startPhoton) m_recorder->startPhoton();
        m_recorder->setBoundaryStatus(GetOpBoundaryProcessStatus());
        m_recorder->RecordStep(step); 
    }

    //if(step_id == m_bounce_max )
    //    track->SetTrackStatus(fStopAndKill);



}

