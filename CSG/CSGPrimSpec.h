#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <vector>
#endif

/**
CSGPrimSpec
=============

Previously assumed that the *sbtIndexOffset* indices 
were global to the entire geometry, but the 2nd-GAS-last-prim-only bug 
suggests that the indices need to be local to each GAS, counting 
from 0 to numPrim-1 for that GAS.

**/

struct CSGPrimSpec
{
    const float*    aabb ; 
    const unsigned* sbtIndexOffset ;   
    unsigned        num_prim ; 
    unsigned        stride_in_bytes ; 
    bool            device ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void downloadDump(const char* msg="CSGPrimSpec::downloadDump") const ; 
    void gather(std::vector<float>& out) const ;
    static void Dump(std::vector<float>& out);
    void dump(const char* msg="CSGPrimSpec::Dump") const ; 
#endif
};

