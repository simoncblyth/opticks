#pragma once

#include "U4_API_EXPORT.hh"
struct U4_API U4OpBoundaryProcess
{
    template <typename T>
    static unsigned GetStatus() ;
};

#include "G4ProcessManager.hh"
#include "InstrumentedG4OpBoundaryProcess.hh"

/**
U4OpBoundaryProcess::GetStatus
--------------------------------

Identifies desired boundary process by dynamic_cast not giving nullptr 

**/

template <typename T>
inline unsigned U4OpBoundaryProcess::GetStatus()
{
    G4OpBoundaryProcessStatus status = Undefined;
    G4ProcessManager* mgr = G4OpticalPhoton::OpticalPhoton()->GetProcessManager() ;
    if(mgr) 
    {
        G4int npmax = mgr->GetPostStepProcessVector()->entries();
        G4ProcessVector* pv = mgr->GetPostStepProcessVector(typeDoIt);
        for (G4int i=0; i<npmax; i++) 
        {
            G4VProcess* proc = (*pv)[i];

            T* opProc = dynamic_cast<T*>(proc);
            if(opProc) 
            { 
                status = opProc->GetStatus(); 
                break;
            }
        }
    }
    return (unsigned)status ; 
}

