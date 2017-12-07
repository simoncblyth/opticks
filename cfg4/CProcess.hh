#pragma once

#include <string>
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

/**
CProcess
============

This only makes sense when called from within stepping.

**/

class G4VProcess ;


struct CFG4_API CProcess
{
    static std::string Desc(G4VProcess* proc) ;  
    static G4VProcess* CurrentProcess() ;  

};

#include "CFG4_TAIL.hh"
 
