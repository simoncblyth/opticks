#pragma once

#include "G4VUserEventInformation.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CEventInfo
============

**/

struct CFG4_API CEventInfo : public G4VUserEventInformation
{
    inline virtual void Print()const{}; 

    CEventInfo( unsigned gencode_ )
       :
       gencode(gencode_)
    {
    }

    unsigned gencode ;  
};

#include "CFG4_TAIL.hh"
