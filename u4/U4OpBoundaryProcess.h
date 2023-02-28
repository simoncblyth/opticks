#pragma once
/**
U4OpBoundaryProcess.h
=======================

* GetStatus identifies the boundary process with type T by trying all optical processes 
  and seeing which one has non-null dynamic_cast

* assumes that the modified boundary process still has method that returns the standard enum::

   G4OpBoundaryProcessStatus GetStatus() const;

**/

#include "U4_API_EXPORT.hh"
struct U4_API U4OpBoundaryProcess
{
    template <typename T>
    static unsigned GetStatus() ;
};

#include "G4ProcessManager.hh"
#include "G4OpBoundaryProcess.hh"   // need standard enum from here even when using customized boundary process


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



#ifdef WITH_PMTSIM
#include "CustomG4OpBoundaryProcess.hh"
template unsigned U4OpBoundaryProcess::GetStatus<CustomG4OpBoundaryProcess>() ; 
#else
#include "InstrumentedG4OpBoundaryProcess.hh"
template unsigned U4OpBoundaryProcess::GetStatus<InstrumentedG4OpBoundaryProcess>() ; 
#endif
