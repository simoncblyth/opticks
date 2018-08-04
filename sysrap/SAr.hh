#pragma once

/**
struct SAr
==============

* seeing issue of stomping on argc/argv with gcc so try capturing it 

**/

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SAr
{
    static SAr* Instance ; 

    SAr( int argc_ , char** argv_ , const char* envvar=0, char delim=' ' ) ;

    void args_from_envvar( const char* envvar, char delim );

    const char* exepath() const ;
    const char* exename() const ;
    static const char* Basename(const char* path);
    std::string argline() const ;
    void dump() const ;

    int    _argc ;
    char** _argv ; 
};



