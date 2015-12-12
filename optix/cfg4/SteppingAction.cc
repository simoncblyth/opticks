#include "SteppingAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4UnitsTable.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"



#include "Recorder.hh"
#include "Format.hh"

#include <cstdio>


SteppingAction::SteppingAction(RecorderBase* recorder)
   : 
   G4UserSteppingAction(),
   m_recorder(recorder)
{ 
  fScintillationCounter = 0;
  fCerenkovCounter      = 0;
  fEventNumber = -1;
}


SteppingAction::~SteppingAction()
{ ; }




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


void SteppingAction::UserSteppingAction(const G4Step* step)
{
  G4int eventNumber = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
  if (eventNumber != fEventNumber) {
   
     G4cout << "SteppingAction::UserSteppingAction evt " << eventNumber << G4endl ;
     fEventNumber = eventNumber;
     fScintillationCounter = 0;
     fCerenkovCounter = 0;
  }


  G4OpBoundaryProcessStatus opStat = GetOpBoundaryProcessStatus();

  //if(opStat != FresnelRefraction)
  G4cout 
        << Format(opStat) 
        << Format(step) 
        << G4endl ; 

  m_recorder->RecordStep(step); 

}

