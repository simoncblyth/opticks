#pragma once
/**
SCurandState.hh
=================

See also qudarap/QCurandState.hh 

Chunked States ?
-------------------

* TODO: require num to be in millions and change num to be expressed in millions with "M" suffix
* TODO: handle chunking by encoding (states_per_chunk, chunk_sequence_index) in the name

  * hmm i recall doing something similar somewhere else with directory naming (probably precooked randoms)
  * want approach to be extendable, adding more chunks without recreating all of them  
  
**/

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



