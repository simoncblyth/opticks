
#include <cstdlib>
#include "PLOG.hh"
#include "SPath.hh"

#include "U4Cerenkov_Debug.hh"
#include "U4Scintillation_Debug.hh"
#include "U4Hit_Debug.hh"

#include "U4Debug.hh"

const plog::Severity U4Debug::LEVEL = PLOG::EnvLevel("U4Debug", "debug" ); 
const char* U4Debug::SaveDir = getenv(EKEY) ;   

const char* U4Debug::GetSaveDir(int eventID)
{
    return SPath::Resolve(SaveDir ? SaveDir : "/tmp" , eventID, DIRPATH );  
}

void U4Debug::EndOfEvent(int eventID)
{
    LOG(LEVEL) << " eventID " << eventID ; 
    U4Cerenkov_Debug::EndOfEvent(eventID);   
    U4Scintillation_Debug::EndOfEvent(eventID);   
    U4Hit_Debug::EndOfEvent(eventID);   
}
