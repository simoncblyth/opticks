#pragma once

/**
QBnd
=====

CUDA-centric equivalent for optixrap/OBndLib 

**/

#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

union quad ; 
struct float4 ; 
struct dim3 ; 

class GBndLib ; 
template <typename T> struct QTex ; 
template <typename T> class NPY ; 

struct QUDARAP_API QBnd
{
    static const plog::Severity LEVEL ;
    static const QBnd*          INSTANCE ; 
    static const QBnd*          Get(); 

    const GBndLib*    blib ; 
    const NPY<double>* dsrc ;  
    const NPY<float>*  src ;  
    QTex<float4>*      tex ; 

    QBnd(const GBndLib* blib); 
    void init(); 
    std::string desc() const ; 

    void makeBoundaryTex(const NPY<float>* buf ) ;
    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    NPY<float>* lookup();
    void lookup( quad* lookup, unsigned num_lookup, unsigned width, unsigned height );
    void dump(   quad* lookup, unsigned num_lookup, unsigned edgeitems=10 );

};


