#pragma once
/**
U4OpBoundaryProcess.h
=======================

* GetStatus identifies the boundary process with type T by trying all optical processes 
  and seeing which one has non-null dynamic_cast

* assumes that the modified boundary process still has method that returns the standard enum::

   G4OpBoundaryProcessStatus GetStatus() const;

Boundary class changes need to match in all the below::

    U4OpBoundaryProcess.h    
    U4Physics.cc 
    U4Recorder.cc
    U4StepPoint.cc


**/

#include "U4_API_EXPORT.hh"
struct U4_API U4OpBoundaryProcess
{
    template <typename T>
    static T* Get() ;

    template <typename T>
    static unsigned GetStatus() ;
};

#include "G4ProcessManager.hh"
#include "G4OpBoundaryProcess.hh"   // need standard enum from here even when using customized boundary process


#if defined(WITH_CUSTOM4) 
#include "C4OpBoundaryProcess.hh"
#elif defined(WITH_PMTSIM) 
#include "CustomG4OpBoundaryProcess.hh"
#elif defined(WITH_INSTRUMENTED_DEBUG)
#include "InstrumentedG4OpBoundaryProcess.hh"
#else
#include "G4OpBoundaryProcess.hh"
#endif


template <typename T>
inline T* U4OpBoundaryProcess::Get()
{
    T* bp = nullptr ; 
    G4ProcessManager* mgr = G4OpticalPhoton::OpticalPhoton()->GetProcessManager() ;
    assert(mgr); 

    G4int pmax = mgr ? mgr->GetPostStepProcessVector()->entries() : 0 ;
    G4ProcessVector* pvec = mgr ? mgr->GetPostStepProcessVector(typeDoIt) : nullptr ;

    for (int i=0; i < pmax ; i++) 
    {
        G4VProcess* p = (*pvec)[i];
        T* t = dynamic_cast<T*>(p);
        if(t) 
        { 
            bp = t ; 
            break;
        }
    }
    return bp ; 
}

template <typename T>
inline unsigned U4OpBoundaryProcess::GetStatus()
{
    T* proc = Get<T>(); 
    G4OpBoundaryProcessStatus status = proc ? proc->GetStatus() : Undefined ;
    return (unsigned)status ; 
}


#if defined(WITH_CUSTOM4)
template unsigned U4OpBoundaryProcess::GetStatus<C4OpBoundaryProcess>() ; 
#elif defined(WITH_PMTSIM)
template unsigned U4OpBoundaryProcess::GetStatus<CustomG4OpBoundaryProcess>() ; 
#elif defined(WITH_INSTRUMENTED_DEBUG)
template unsigned U4OpBoundaryProcess::GetStatus<InstrumentedG4OpBoundaryProcess>() ; 
#else
template unsigned U4OpBoundaryProcess::GetStatus<G4OpBoundaryProcess>() ; 
#endif

