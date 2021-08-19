#pragma once

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct dim3 ; 
struct NP ; 
template <typename T> struct QTex ; 

struct QUDARAP_API QScint
{
    static const plog::Severity LEVEL ; 
    static const QScint*        INSTANCE ; 
    static const QScint*        Get(); 

    const NP*      dsrc ; 
    const NP*      src ; 
    QTex<float>*    tex ; 

    QScint(const NP* icdf, unsigned hd_factor); 

    void init(); 
    static QTex<float>* MakeScintTex(const NP* src, unsigned hd_factor);
    std::string desc() const ; 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void check();
    NP*  lookup();

    void lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height ); 
    void dump(   float* lookup, unsigned num_lookup, unsigned edgeitems=10 ); 

};


