#pragma once

/**
QTexLookup
===========

This provides a basic test for a GPU texture, looking 
up every texel of the texture.  This resulting array can be compared
with the input array and should exactly match.

**/

#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct NP ; 
struct dim3 ; 
template<typename T> struct QTex ; 

template<typename T>
struct QUDARAP_API QTexLookup
{
    static const plog::Severity LEVEL ; 
    static NP* Look(const  QTex<T>* tex_); 

    QTexLookup( const QTex<T>* tex_ ); 
    const QTex<T>* tex ;  

    NP* lookup();
    void lookup_( T* lookup, unsigned num_lookup, unsigned width, unsigned height  ); 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height ); 

}; 




