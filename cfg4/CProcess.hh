#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CProcess
============

This only makes sense when called from within stepping.

**/

class G4VProcess ;

class CFG4_API CProcess 
{
    public:
        static G4VProcess* CurrentProcess() ; 
};

#include "CFG4_TAIL.hh"
 
