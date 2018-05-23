#pragma once

struct BConfig ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API NLODConfig 
{
    static const char* instanced_lodify_onload_ ; 

    NLODConfig(const char* cfg);
    struct BConfig* bconfig ;  
    void dump(const char* msg="NLODConfig::dump") const ; 

    int verbosity ; 
    int levels ; 
    int instanced_lodify_onload ; 
};

