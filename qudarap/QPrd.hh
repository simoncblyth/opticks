#pragma once

#include "QUDARAP_API_EXPORT.hh"
#include <vector>
struct float4 ; 
struct quad2 ; 
struct QBnd ; 

/**
QPrd
=====

MUST be instanciated after QBnd 

Dummy per-ray-data intended for pure-CUDA (no OptiX) testing 
of propagation using QSimTest MOCK_PROPAGATE 

**/

struct QUDARAP_API QPrd
{
    static const QPrd* INSTANCE  ; 
    static const QPrd* Get() ; 

    const QBnd* bnd ; 

    std::vector<unsigned> bnd_idx ; 
    std::vector<float4> nrmt ; 
    std::vector<quad2>  prd ; 

    QPrd(); 
    void init();   
    void dump(const char* msg="QPrd::dump") const ; 

};



