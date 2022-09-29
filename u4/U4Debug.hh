#pragma once

/**
U4Debug.hh
=============

U4Cerenkov_Debug
   records scintillation steps, 

U4Scintillation_Debug
   records cerenkov steps, eg for investigating a lack of gensteps 

U4Hit_Debug
   records labels of hits with gs indices


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
    static constexpr const char* EKEY = "U4Debug_SaveDir" ;   
    static const char* SaveDir ; 

    static const char* GetSaveDir(int eventID); 
    static void Save(int eventID); 
};




