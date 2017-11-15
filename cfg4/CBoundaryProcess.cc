
#include "G4ProcessManager.hh"


#include "CBoundaryProcess.hh"


#ifdef USE_CUSTOM_BOUNDARY
DsG4OpBoundaryProcessStatus CBoundaryProcess::GetOpBoundaryProcessStatus()
{
    DsG4OpBoundaryProcessStatus status = Undefined;
#else
G4OpBoundaryProcessStatus CBoundaryProcess::GetOpBoundaryProcessStatus()
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


