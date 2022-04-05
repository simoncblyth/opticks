#pragma once

/**
QBnd
=====

CUDA-centric equivalent for optixrap/OBndLib 

**/

#include <vector>
#include <string>
#include "plog/Severity.h"
#include "QUDARAP_API_EXPORT.hh"

union quad ; 
struct float4 ; 
struct dim3 ; 

template <typename T> struct QTex ; 
struct NP ; 

struct QUDARAP_API QBnd
{
    static const plog::Severity LEVEL ;
    static const unsigned       MISSING ; 
    static const QBnd*          INSTANCE ; 
    static const QBnd*          Get(); 

    std::vector<std::string>  bnames ; 
    const NP*      dsrc ;  
    const NP*      src ;  
    QTex<float4>*  tex ; 

    QBnd(const NP* buf); 

    std::string descBoundary() const ;
    std::string desc() const ; 

    unsigned getNumBoundary() const ; 
    const char* getBoundarySpec(unsigned idx) const ; 
    void        getBoundarySpec(std::vector<std::string>& names, const unsigned* idx , unsigned num_idx ) const ; 

    unsigned    getBoundaryIndex(const char* spec) const ;

    void        getBoundaryIndices( std::vector<unsigned>& bnd_idx, const char* bnd_sequence, char delim=',' ) const ; 
    void        dumpBoundaryIndices( const std::vector<unsigned>& bnd_idx ) const ; 

    unsigned    getBoundaryLine(const char* spec, unsigned j) const ; 
    unsigned    getMaterialLine( const char* material ) const ; 

    static QTex<float4>* MakeBoundaryTex(const NP* buf ) ;
    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    NP*  lookup();
    void lookup( quad* lookup, unsigned num_lookup, unsigned width, unsigned height );
    void dump(   quad* lookup, unsigned num_lookup, unsigned edgeitems=10 );

};


