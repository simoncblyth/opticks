#pragma once

/**
QProp
=========

See ~/np/tests/NPInterp.py 

TODO:

1. constructing compound prop array from many indiviual prop arrays 
   with various domain lengths and differing domain values 
   (for now only really need for RINDEX for LS,Water,Acrylic 
   but keep it flexible : needed for Cerenkov generation matching)


**/

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

union quad ; 
struct float4 ; 
struct dim3 ; 

struct NP ; 

struct QUDARAP_API QProp
{
    static const plog::Severity LEVEL ;
    static const QProp*       INSTANCE ; 
    static const QProp*       Get(); 

    const NP* prop  ;  
    const float* pp ; 
    unsigned nv ; 
    unsigned ni ; 
    unsigned nj ; 

    float* d_pp ; 

    QProp(const NP* prop_); 

    void init(); 
    void dump(); 
    void upload(); 
    void clear(); 

    void lookup(float x0, float x1, unsigned nx);
    void lookup( float* lookup, const float* domain,  unsigned lookup_prop, unsigned domain_width ); 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

};


