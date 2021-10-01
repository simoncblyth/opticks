#pragma once

#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct NP ; 
struct dim3 ; 
template<typename T> struct QTex ; 

template<typename T>
struct QUDARAP_API QTexLookup
{
    static const plog::Severity LEVEL ; 

    QTexLookup( const QTex<T>* tex_ ); 
    const QTex<T>* tex ;  

    NP* lookup();
    void lookup_( T* lookup, unsigned num_lookup, unsigned width, unsigned height  ); 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height ); 

}; 




