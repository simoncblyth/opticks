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
struct qprop ; 

struct NP ; 

struct QUDARAP_API QProp
{
    static const plog::Severity LEVEL ;
    static const QProp*       INSTANCE ; 
    static const QProp*       Get(); 

    const NP* a  ;  
    const float* pp ; 
    unsigned nv ; 
    unsigned ni ; 
    unsigned nj ; 

    qprop* prop ; 
    qprop* d_prop ; 

    QProp(const NP* a_); 

    void init(); 
    void dump(); 
    void uploadProps(); 

    void lookup( float* lookup, const float* domain,  unsigned lookup_prop, unsigned domain_width ); 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

};


