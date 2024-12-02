#pragma once
/**
SCurandStateMonolithic.hh
==========================

See also qudarap/QCurandState.hh 


**/

#include <string>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SCurandStateMonolithic
{
    typedef unsigned long long ULL ; 
    static constexpr const ULL M = 1000000ull ; 

    static const plog::Severity LEVEL ; 
    static const char* RNGDIR ;
    static const char* NAME_PREFIX ; 
    static const char* DEFAULT_PATH ; 

    static std::string Desc() ;  
    static const char* Path() ; 
    static std::string Stem_(ULL num, ULL seed, ULL offset); 
    static std::string Path_(ULL num, ULL seed, ULL offset); 
    static long RngMax() ; 
    static long RngMax(const char* path) ; 


    SCurandStateMonolithic(const char* spec); 
    SCurandStateMonolithic(ULL num, ULL seed, ULL offset) ; 
    void init(); 

    std::string desc() const ; 

    const char* spec ; 
    ULL num    ; 
    ULL seed   ; 
    ULL offset ;  
    std::string path ; 
    bool exists ; 
    long rngmax ; 

};



