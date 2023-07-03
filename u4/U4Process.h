#pragma once

/**
U4Process.h : Cherry Picking from cfg4/CProcessManager
================================================================

Process identification relies on the processes using 
their default names, for example from "g4-cls G4OpRayleigh"
the default process name is "OpRayleigh". 

**/

#include <cassert>
#include <cstring>
#include <string>
#include <sstream>

#include "G4ProcessManager.hh"
#include "G4VDiscreteProcess.hh"
#include "G4OpticalPhoton.hh"
#include "G4Track.hh"
#include "G4Step.hh"

enum {
   U4Process_Unknown        = 0, 
   U4Process_Transportation = 1, 
   U4Process_Scintillation  = 2,
   U4Process_OpAbsorption   = 3,
   U4Process_OpRayleigh     = 4,
   U4Process_OpBoundary     = 5
}; 

struct U4Process
{
    static constexpr const char* Transportation_ = "Transportation" ; 
    static constexpr const char* Scintillation_ = "Scintillation" ; 
    static constexpr const char* OpAbsorption_ = "OpAbsorption" ; 
    static constexpr const char* OpRayleigh_ = "OpRayleigh" ; 
    static constexpr const char* OpBoundary_ = "OpBoundary" ; 

    template<typename T> static T* Get() ; 

    static const char* Name(const G4VProcess* proc); 
    static unsigned ProcessType(const char* name); 
    static bool IsKnownProcess(unsigned type); 
    static bool IsNormalProcess(unsigned type); 
    static bool IsPeculiarProcess(unsigned type); 

    static G4ProcessManager* GetManager(); 
    static std::string Desc(); 

    static void ClearNumberOfInteractionLengthLeft(const G4Track& aTrack, const G4Step& aStep); 

};

/**
U4Process::Get
----------------

Find a process of the OpticalPhoton process manager 
with the template type. If none of the processes 
are able to be dynamically cast to the template 
type then nullptr is returned. Usage::

     G4OpRayleigh* p = U4Process::Get<G4OpRayleigh>() ; 


**/

template<typename T>
inline T* U4Process::Get()
{
    G4ProcessManager* mgr = GetManager(); 
    if(mgr == nullptr)
    {
        std::cerr << "U4Process::Get FAILED : MAYBE DO THIS LATER " << std::endl ; 
        return nullptr ; 
    }
    G4ProcessVector* procv = mgr->GetProcessList() ;
    G4int n = procv->entries() ;

    T* p = nullptr ; 
    int count = 0 ; 

    for(int i=0 ; i < n ; i++)
    {   
        G4VProcess* proc = (*procv)[i] ; 
        T* proc_ = dynamic_cast<T*>(proc) ; 
        if(proc_ != nullptr)
        {
            p = proc_ ; 
            count += 1 ; 
        }
    }
    assert( count == 0 || count == 1 ); 
    return p ; 
}


inline const char* U4Process::Name(const G4VProcess* proc)
{
    const G4String& name = proc->GetProcessName() ;
    return name.c_str(); 
}
inline unsigned U4Process::ProcessType(const char* name)
{
    unsigned type = U4Process_Unknown ; 
    if(strcmp(name, Transportation_) == 0)  type = U4Process_Transportation ; 
    if(strcmp(name, Scintillation_) == 0)   type = U4Process_Scintillation ; 
    if(strcmp(name, OpAbsorption_) == 0)    type = U4Process_OpAbsorption ; 
    if(strcmp(name, OpRayleigh_) == 0)      type = U4Process_OpRayleigh ; 
    if(strcmp(name, OpBoundary_) == 0)      type = U4Process_OpBoundary ; 
    return type ;  
}

inline bool U4Process::IsKnownProcess(unsigned type)
{ 
     return type != U4Process_Unknown ; 
}
inline bool U4Process::IsNormalProcess(unsigned type)
{
    return type == U4Process_OpRayleigh || type == U4Process_OpAbsorption ; 
} 
inline bool U4Process::IsPeculiarProcess(unsigned type)
{
    return type == U4Process_Transportation || type == U4Process_Scintillation || type == U4Process_OpBoundary ; 
} 
inline G4ProcessManager* U4Process::GetManager()
{
    return G4OpticalPhoton::OpticalPhotonDefinition()->GetProcessManager() ; 
}

std::string U4Process::Desc()
{
    G4ProcessManager* mgr = GetManager(); 
    assert(mgr); 
    std::stringstream ss ; 
    ss << "U4Process::Desc" ;
    G4ProcessVector* procv = mgr->GetProcessList() ;
    G4int n = procv->entries() ;
    ss << " n:[" << n << "]" << std::endl ;   
    for(int i=0 ; i < n ; i++)
    {   
        G4VProcess* proc = (*procv)[i] ; 
        const char* name = Name(proc);  
        unsigned type = ProcessType(name); 
        assert(IsKnownProcess(type)); 
        bool peculiar = IsPeculiarProcess(type); 
        G4double lenleft = proc->GetNumberOfInteractionLengthLeft() ; 
        bool unset = lenleft == -1. ; 
        bool expected = peculiar == unset ; 
        //assert( peculiar == unset );  
        // not obeyed when called from U4Recorder::PreUserTrackingAction_Optical but is from stepping

        ss << " (" << i << ")" 
           << " name " << name 
           << " type " << type
           << " lenleft " << lenleft
           << " unset " << unset
           << " peculiar " << peculiar 
           << " expected " << ( expected ? "YES" : "NO" )
           << std::endl
           ;   
    }   
    return ss.str();
}


/**
U4Process::ClearNumberOfInteractionLengthLeft
-----------------------------------------------

This clears the interaction length left for "normal" optical processes : OpAbsorption and OpRayleigh 
with no use of G4UniformRand.

Formerly canonically called from tail of CManager::UserSteppingAction/CManager::prepareForNextStep
when track is not terminating (ie not track status fStopAndKill so there is another step).

The effect of this is to make the Geant4 random consumption more regular which 
makes it easier to align with Opticks random consumption. 

This provides a devious way to invoke the protected ClearNumberOfInteractionLengthLeft 
via the public G4VDiscreteProcess::PostStepDoIt

g4-;g4-cls G4VDiscreteProcess::

    112 G4VParticleChange* G4VDiscreteProcess::PostStepDoIt(
    113                             const G4Track& ,
    114                             const G4Step&
    115                             )
    116 { 
    117 //  clear NumberOfInteractionLengthLeft
    118     ClearNumberOfInteractionLengthLeft();
    119 
    120     return pParticleChange;
    121 }


Noticed that returns from::

     InstrumentedG4OpBoundaryProcess::PostStepDoIt
     G4OpAbsorption::PostStepDoIt
     G4OpRayleigh::PostStepDoIt

All do G4VDiscreteProcess::PostStepDoIt, eg::

     182 G4VParticleChange*
     183 InstrumentedG4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     184 {
     ...
     623         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     624 }


Q: Why the artificial call when the PostStepDoIt returns do it anyhow ?
A: Because only the winning process PostStepDoIt runs, and that changes 
   with the history : so you get a complicated consumption history 
   that would be difficult to align with.  

**/

void U4Process::ClearNumberOfInteractionLengthLeft(const G4Track& aTrack, const G4Step& aStep)
{
    G4ProcessManager* mgr = GetManager(); 
    G4ProcessVector* procv = mgr->GetProcessList() ;
    for(int i=0 ; i < int(procv->entries()) ; i++)
    {
        G4VProcess* proc = (*procv)[i] ;
        unsigned type = ProcessType(Name(proc)) ;   
        if(IsNormalProcess(type))
        {
            G4VDiscreteProcess* dproc = dynamic_cast<G4VDiscreteProcess*>(proc) ;
            assert(dproc);
            dproc->G4VDiscreteProcess::PostStepDoIt( aTrack, aStep );
        }
    }
}



