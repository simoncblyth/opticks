#pragma once

#include <string>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SCurandState
{
    static const plog::Severity LEVEL ; 
    static const char* RNGDIR ;
    static const char* NAME_PREFIX ; 
    static const char* DEFAULT_PATH ; 

    static std::string Desc() ;  
    static const char* Path() ; 
    static std::string Stem_(unsigned long long num, unsigned long long seed, unsigned long long offset); 
    static std::string Path_(unsigned long long num, unsigned long long seed, unsigned long long offset); 
    static long RngMax() ; 
    static long RngMax(const char* path) ; 


    SCurandState(const char* spec); 
    SCurandState(unsigned long long num, unsigned long long seed, unsigned long long offset) ; 
    void init(); 

    std::string desc() const ; 

    const char* spec ; 
    unsigned long long num    ; 
    unsigned long long seed   ; 
    unsigned long long offset ;  
    std::string path ; 
    bool exists ; 
    long rngmax ; 

};



