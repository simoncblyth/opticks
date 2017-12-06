#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CStepping
============

This only makes sense when called from within stepping.

**/

#include "CSteppingState.hh"

class CFG4_API CStepping
{
    public:
        static CSteppingState CurrentState() ; 
};

#include "CFG4_TAIL.hh"
 
