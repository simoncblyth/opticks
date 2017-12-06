
#include "G4EventManager.hh"
#include "G4TrackingManager.hh"
#include "G4SteppingManager.hh"

#include "CStepping.hh"


CSteppingState CStepping::CurrentState()
{
    G4EventManager* evtMgr = G4EventManager::GetEventManager() ;
    G4TrackingManager* trkMgr = evtMgr->GetTrackingManager() ; 
    G4SteppingManager* stepMgr = trkMgr->GetSteppingManager() ; 
 
    CSteppingState ss ; 
    ss.fPostStepGetPhysIntVector = stepMgr->GetfPostStepGetPhysIntVector();
    //ss.fSelectedPostStepDoItVector = stepMgr->GetfSelectedPostStepDoItVector() ;
    ss.fCurrentProcess = stepMgr->GetfCurrentProcess() ; 
    ss.MAXofPostStepLoops = stepMgr->GetMAXofPostStepLoops() ;
    ss.fStepStatus        = stepMgr->GetfStepStatus() ; 
    
    return ss ; 
}



