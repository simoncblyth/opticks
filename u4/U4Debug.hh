#pragma once

/**
U4Debug.hh : Umbrella struct for coordinated saving of debug arrays 
=====================================================================

The below debug structs were developed to investigate issues such as a lack of gensteps.  
As these structs are used purely for debug, not monitoring, usage is typically 
hidden behind the WITH_G4CXOPTICKS_DEBUG preprocessor macro, which is not normally active. 

U4Cerenkov_Debug
   records detailed debug info for cerenkov steps 
   collected for example from G4Cerenkov_modified::PostStepDoIt

U4Scintillation_Debug
   records detailed debug info for scintillation steps
   collected for example from DsG4Scintillation::PostStepDoIt shortly before the

U4Hit_Debug
   records photon spho labels of hits with gs indices
   collected for example from junoSD_PMT_v2::SaveNormHit/junoSD_PMT_v2::ProcessHits  


To make the connection between the debug steps and labels ? 
Is not so simple because the purposes are different.  
Want to record steps that yield no gensteps in order to 
understand lack of gensteps. 

**/


#include "plog/Severity.h"
#include "U4_API_EXPORT.hh"

struct U4_API U4Debug
{   
    static const plog::Severity LEVEL ; 
    static constexpr const char* ETOK = "${U4Debug_SaveDir:-$TMP}/U4Debug" ;  // see spath::ResolveToken
    //static const char* SaveDir ; 
    //static const char* GetSaveDir(int eventID); 

    static void Save(int eventID); 
};




