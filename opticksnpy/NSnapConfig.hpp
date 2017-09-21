#pragma once

struct BConfig ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API NSnapConfig 
{
    NSnapConfig(const char* cfg);
    struct BConfig* bconfig ;  
    void dump(const char* msg="NSnapConfig::dump") const ; 

    int verbosity ; 
    int steps ; 
    float eyestartz ; 
    float eyestopz ; 
};

