#pragma once
#include <cstddef>

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SId 
{
    SId(const char* identifiers_);  
    const char* get(bool reset=false); 

    const char* identifiers ; 
    int idx ; 
};


