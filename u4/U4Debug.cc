
#include <cstdlib>
#include "SLOG.hh"
#include "SPath.hh"
#include "SEvt.hh"

#include "U4Cerenkov_Debug.hh"
#include "U4Scintillation_Debug.hh"
#include "U4Hit_Debug.hh"

#include "U4Debug.hh"

const plog::Severity U4Debug::LEVEL = SLOG::EnvLevel("U4Debug", "DEBUG" ); 
const char* U4Debug::SaveDir = getenv(EKEY) ;   

const char* U4Debug::GetSaveDir(int eventID)
{
    return SPath::Resolve(SaveDir ? SaveDir : "/tmp" , eventID, DIRPATH );  
}

/**
U4Debug::Save
---------------

This is used for example from junoSD_PMT_v2::EndOfEvent WITH_G4CXOPTICKS_DEBUG

**/

void U4Debug::Save(int eventID)
{
    const char* dir = GetSaveDir(eventID); 
    std::cout 
        << "U4Debug::Save"  
        << " eventID " << eventID 
        << " dir " << dir 
        << " EKEY " << EKEY 
        << std::endl
        ; 

    LOG(LEVEL) 
        << " eventID " << eventID 
        << " dir " << dir 
        << " EKEY " << EKEY 
        ; 

    U4Cerenkov_Debug::Save(dir);   
    U4Scintillation_Debug::Save(dir);   
    U4Hit_Debug::Save(dir);   

    SEvt::SaveGenstepLabels( dir ); 
}
