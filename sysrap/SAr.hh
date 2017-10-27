#pragma once

/**
struct SAr
==============

* seeing issue of stomping on argc/argv with gcc so try capturing it 

**/

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SAr
{
    SAr( int argc_ , char** argv_ ) ;
    void dump();

    int    _argc ;
    char** _argv ; 
};



