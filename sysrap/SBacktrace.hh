#pragma once

#include "SYSRAP_API_EXPORT.hh"
#include <ostream>

struct SYSRAP_API SBacktrace
{
    static void Dump(); 
    static void Dump(std::ostream& out) ;
    static const char* CallSite(const char* call="::flat()" , bool addr=true );  
};



