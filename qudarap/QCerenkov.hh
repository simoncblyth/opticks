#pragma once

/**
QCerenkov
===========

Loads icdf and creates GPU texture from it ready for energy sampling. 

See also:

QCerenkovIntegral 
    time consuming integration of RINDEX s2 to form the icdf, to be done pre-cache
QCK
    persisting QCerenkovIntegral results 
ana/rindex.py
    prototyping/experimentation

**/

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct float4 ; 
struct NP ; 
template <typename T> struct QTex ; 
template <typename T> struct QTexLookup ; 
struct dim3 ; 

struct QUDARAP_API QCerenkov
{
    static const plog::Severity LEVEL ; 
    static const QCerenkov*     INSTANCE ; 
    static const QCerenkov*     Get(); 
    static const char*          DEFAULT_FOLD ; 
    static NP*                  Load(const char* fold, const char* name) ; 
    static QTex<float4>*        MakeTex(const NP* icdf, char filterMode, bool normalizedCoords) ; 

    const char*             fold ; 
    const NP*               icdf_ ; 
    const NP*               icdf ; 
    char                    filterMode ; 
    bool                    normalizedCoords ; 
    QTex<float4>*           tex ; 
    QTexLookup<float4>*     look ; 

    QCerenkov(const char* fold=nullptr); 
    void init(); 
    std::string desc() const ; 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void check();
    NP* lookup();
    void dump(   float* lookup, unsigned num_lookup, unsigned edgeitems=10 ); 

};


