#pragma once

/**
SFrame
========

Used for stack frame introspection based on *cxxabi.h*

**/


#include "SYSRAP_API_EXPORT.hh"
#include <ostream>

struct SYSRAP_API SFrame
{
    SFrame( char* line ) ;
    ~SFrame();

    void parse();
    char* demangle(); // fails for non C++ symbols
    void dump();
    void dump(std::ostream& out);

    char* line ; 
    char* name ; 
    char* offset ;
    char* end_offset ;
 
    char* func ;    // only func is "owned"
};



