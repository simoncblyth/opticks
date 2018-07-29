#pragma once
#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

/**
SId
====

Supply single char identifiers from ctor argument string, 
until run out at which point cycle is incremented so give
an integer suffix. 

reset returns to the first identifier.

This is used for code generation in X4Solid, search for g4code.
 
**/


struct SYSRAP_API SId 
{
    SId(const char* identifiers_);  

    const char* get(bool reset=false); 
    void reset(); 

    const char* identifiers ; 
    int         len ; 
    int         idx ; 
    int         cycle ; 
};


