#pragma once

struct BConfig ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API NLODConfig 
{
    NLODConfig(const char* cfg);
    struct BConfig* bconfig ;  
    void dump(const char* msg="NLODConfig::dump") const ; 

    int verbosity ; 
    int levels ; 
};

