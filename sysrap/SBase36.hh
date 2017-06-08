#pragma once

#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SBase36
{
    static const char* LABELS ;  
    std::string operator()(unsigned int val) const ; 
};





