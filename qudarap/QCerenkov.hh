#pragma once

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct NP ; 
template <typename T> struct QTex ; 
struct dim3 ; 

/**
QCerenkov
===========

Prototyping/experimentation done in ana/rindex.py 

**/

struct QUDARAP_API QCerenkov
{
    static const plog::Severity LEVEL ; 
    static const QCerenkov*        INSTANCE ; 
    static const QCerenkov*        Get(); 
    static const char* DEFAULT_PATH ; 
    static NP* Load(const char* path_) ; 

    const char*             path ; 
    NP*                     dsrc ; 
    NP*                     src ; 
    QTex<float>*            tex ; 

    QCerenkov(const char* path=nullptr); 

    void init(); 
    void makeTex(const NP* dsrc);
    std::string desc() const ; 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void check();
    NP* lookup();
    void lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height ); 
    void dump(   float* lookup, unsigned num_lookup, unsigned edgeitems=10 ); 

};


