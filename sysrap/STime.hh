#pragma once

#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API STime 
{
    static int EpochSeconds(); 
    static std::string Format(const char* fmt="%Y-%m-%d.%X", int epochseconds=0 ); 
};


