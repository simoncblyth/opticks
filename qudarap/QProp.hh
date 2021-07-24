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
    static const char*  DEFAULT_PATH ;
    static const QProp*       INSTANCE ; 
    static const QProp*       Get(); 

    const char* path ; 
    const NP* a  ;  
    const float* pp ; 
    unsigned nv ; 

    unsigned ni ; 
    unsigned nj ; 
    unsigned nk ; 

    qprop* prop ; 
    qprop* d_prop ; 

    static const NP* Load(const char* path_ ); 

    QProp(const char* path=nullptr); 
    virtual ~QProp(); 
    void init(); 
    void uploadProps(); 

    void dump() const ; 
    std::string desc() const ;
    qprop* getDevicePtr() const ;
    void lookup( float* lookup, const float* domain,  unsigned lookup_prop, unsigned domain_width ) const ; 
    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height ) const ;
};


