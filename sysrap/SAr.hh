#pragma once

/**
struct SAr
==============

* seeing issue of stomping on argc/argv with gcc so try capturing it 

**/

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SAr
{
    SAr( int argc_ , char** argv_ , const char* envvar=0, char delim=' ' ) ;

    void args_from_envvar( const char* envvar, char delim );
    void dump();

    int    _argc ;
    char** _argv ; 
};



