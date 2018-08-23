#pragma once
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SFrame
{
    SFrame( char* line ) ;
    ~SFrame();

    void parse();
    char* demangle(); // fails for non C++ symbols
    void dump();

    char* line ; 
    char* name ; 
    char* offset ;
    char* end_offset ;
 
    char* func ;    // only func is "owned"
};



