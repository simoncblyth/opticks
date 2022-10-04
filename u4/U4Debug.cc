
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

void U4Debug::Save(int eventID)
{
    const char* dir = GetSaveDir(eventID); 
    LOG(LEVEL) << " eventID " << eventID << " dir " << dir ; 

    U4Cerenkov_Debug::Save(dir);   
    U4Scintillation_Debug::Save(dir);   
    U4Hit_Debug::Save(dir);   

    SEvt::SaveGenstepLabels( dir ); 
}
