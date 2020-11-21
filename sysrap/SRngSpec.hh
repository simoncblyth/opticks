#pragma once
/**
SRngSpec
==================

**/

#include <cstddef>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

class SYSRAP_API SRngSpec
{
    public:
        static const plog::Severity LEVEL ; 
        static const char* DefaultRngDir() ; 
        static const char* CURANDStatePath(const char* rngdir=NULL, unsigned rngmax=3000000, unsigned long long seed=0, unsigned long long offset=0 ); 
    public:
        SRngSpec(unsigned rngmax, unsigned long long seed, unsigned long long offset);

        const char* getCURANDStatePath(const char* rngdir=NULL) const ;  
        bool        isValid(const char* rngdir=NULL) const ;
        std::string desc() const ;
    private:
        unsigned           m_rngmax ; 
        unsigned long long m_seed ; 
        unsigned long long m_offset ; 
}; 


