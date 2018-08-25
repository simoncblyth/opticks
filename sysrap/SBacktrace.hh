#pragma once

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SBacktrace
{
    static void Dump(); 
    static const char* CallSite(const char* call="::flat()" , bool addr=true );  
};



