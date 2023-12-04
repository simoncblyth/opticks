
#include <cstdlib>
#include "SLOG.hh"

#include "spath.h"
#include "sstr.h"

#include "SEvt.hh"

#include "U4Cerenkov_Debug.hh"
#include "U4Scintillation_Debug.hh"
#include "U4Hit_Debug.hh"

#include "U4Debug.hh"

const plog::Severity U4Debug::LEVEL = SLOG::EnvLevel("U4Debug", "DEBUG" ); 



/**
U4Debug::Save
---------------

This is used for example from junoSD_PMT_v2::EndOfEvent WITH_G4CXOPTICKS_DEBUG

**/

void U4Debug::Save(int eventID)
{
    const char* dir = spath::Resolve( ETOK, sstr::FormatIndex(eventID, 'B', 3, nullptr) ); 

    LOG(LEVEL) 
        << " eventID " << eventID 
        << " dir " << dir 
        << " ETOK " << ETOK 
        ; 

    U4Cerenkov_Debug::Save(dir);   
    U4Scintillation_Debug::Save(dir);   
    U4Hit_Debug::Save(dir);   

    SEvt::SaveGenstepLabels( dir ); 
}
