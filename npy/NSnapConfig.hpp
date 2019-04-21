#pragma once

#include <string>
#include "plog/Severity.h"

struct BConfig ; 

#include "NPY_API_EXPORT.hh"

/**
NSnapConfig
============

Principal consumer is OpTracer::snap


**/



struct NPY_API NSnapConfig 
{
    static const plog::Severity LEVEL ; 

    NSnapConfig(const char* cfg);
    BConfig* bconfig ;  
    void dump(const char* msg="NSnapConfig::dump") const ; 

    int verbosity ; 
    int steps ; 
    int fmtwidth ; 
    float eyestartz ; 
    float eyestopz ; 
    std::string prefix ; 
    std::string postfix ; 


    std::string getSnapPath(unsigned index);
    static std::string SnapIndex(unsigned index, unsigned width);
    std::string desc() const ; 



};

