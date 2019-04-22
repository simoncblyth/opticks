#pragma once

#include <string>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API STime 
{
    static const char* FMT ;   // 
    static int EpochSeconds(); 
    static std::string Format(int epochseconds=0, const char* fmt=NULL ); 
};


