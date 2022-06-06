#include "G4ProcessManager.hh"
#include "G4OpBoundaryProcess.hh"
#include "U4OpBoundaryProcess.hh"

unsigned U4OpBoundaryProcess::GetStatus()
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
            // NULL casts expected, that is the way of finding the wanted boundary process
        }
    }
    return (unsigned)status ; 
}




