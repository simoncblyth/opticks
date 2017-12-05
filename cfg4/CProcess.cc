

#include "G4EventManager.hh"
#include "G4TrackingManager.hh"
#include "G4SteppingManager.hh"

#include "CProcess.hh"

G4VProcess* CProcess::CurrentProcess()
{
    G4EventManager* evtMgr = G4EventManager::GetEventManager() ;
    G4TrackingManager* trkMgr = evtMgr->GetTrackingManager() ; 
    G4SteppingManager* stepMgr = trkMgr->GetSteppingManager() ; 
    return stepMgr->GetfCurrentProcess() ; 
}


