#pragma once

/**
QCerenkov : using ICDF lookup
=================================

Normally Cerenkov is simulated using rejection sampling, BUT that 
is problematic on GPU as the rejection sampling of RINDEX is 
very sensitive to the use of float or double. 

Lookup sampling from ICDF is less sensitive. 

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
struct qcerenkov ; 

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

    static qcerenkov* MakeInstance(); 

    const char*             fold ; 
    const NP*               icdf_ ; 
    const NP*               icdf ; 
    char                    filterMode ; 
    bool                    normalizedCoords ; 
    QTex<float4>*           tex ; 
    QTexLookup<float4>*     look ; 
    qcerenkov*              cerenkov ; 
    qcerenkov*              d_cerenkov ; 

    QCerenkov(); 
    QCerenkov(const char* fold); 
    void init(); 
    std::string desc() const ; 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void check();
    NP* lookup();
    void dump(   float* lookup, unsigned num_lookup, unsigned edgeitems=10 ); 

};


