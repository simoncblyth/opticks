#pragma once

/**
struct SAr
==============

**/

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SAr
{
    static SAr* Instance ; 

    SAr( int argc_ , char** argv_ , const char* envvar=0, char delim=' ' ) ;

    void args_from_envvar( const char* envvar, char delim );
    void sanitycheck() const ; 

    const char* exepath() const ;
    const char* exename() const ;
    static const char* Basename(const char* path);
    std::string argline() const ;
    const char* cmdline() const ;
    const char* get_arg_after(const char* arg, const char* fallback) const ;
    bool has_arg( const char* arg ) const ; 
    void dump() const ;

    int    _argc ;
    char** _argv ; 

    const char* _cmdline ; 
};



