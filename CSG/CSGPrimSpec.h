#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <vector>
#include <string>
#include "CSG_API_EXPORT.hh"
#endif

/**
CSGPrimSpec
=============

* *CSGPrimSpec* provides the specification to access the AABB and sbtIndexOffset of all CSGPrim of a CSGSolid.  
* The specification includes pointers, counts and stride.
* Instances are created for a solidIdx by CSGFoundry::getPrimSpec using CSGPrim::MakeSpec
* Crucially *CSGPrimSpec* is used to pass the AABB for a solid to CSGOptix/GAS_Builder.

Previously assumed that the *sbtIndexOffset* indices were global 
to the entire geometry, but the 2nd-GAS-last-prim-only bug indicates 
that the indices need to be local to each GAS, counting 
from 0 to numPrim-1 for that GAS.

**/

struct CSG_API CSGPrimSpec
{
    const float*    aabb ; 
    const unsigned* sbtIndexOffset ;   
    unsigned        num_prim ; 
    unsigned        stride_in_bytes ; 
    bool            device ; 
    unsigned        primitiveIndexOffset ;   // offsets optixGetPrimitiveIndex() see GAS_Builder::MakeCustomPrimitivesBI_11N

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void downloadDump(const char* msg="CSGPrimSpec::downloadDump") const ; 
    void gather(std::vector<float>& out) const ;
    static void Dump(std::vector<float>& out);
    void dump(const char* msg="CSGPrimSpec::dump") const ; 
    std::string desc() const ; 
#endif
};

