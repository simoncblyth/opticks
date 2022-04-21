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


    static std::string DescOptical(const NP* optical, const NP* bnd ); 
    static void Add( NP** opticalplus, NP** bndplus, const NP* optical, const NP* bnd,  const std::vector<std::string>& specs ); 

    static void GetOpticalValues( uint4& item, unsigned i, unsigned j, const char* qname ); 
    static NP*  AddOptical( const NP* optical, const std::vector<std::string>& bnames, const std::vector<std::string>& specs ) ; 
    static void GetPerfectValues( std::vector<float>& values, unsigned nk, unsigned nl, unsigned nm, const char* name ); 
    static NP*  AddBoundary( const NP* src, const char* specs, char delim='\n' ); 
    static NP*  AddBoundary( const NP* src, const std::vector<std::string>& specs ); 
    static std::string DescDigest(const NP* bnd, int w=16) ; 

    static bool FindName( unsigned& i, unsigned& j, const char* qname, const std::vector<std::string>& names ); 
    bool   findName( unsigned& i, unsigned& j, const char* qname ) const ; 

    std::vector<std::string>  bnames ; 
    const NP*      dsrc ;  
    const NP*      src ;  
    QTex<float4>*  tex ; 

    static const NP* NarrowIfWide(const NP* buf ); 
    QBnd(const NP* buf); 

    std::string getItemDigest( int i, int j, int w=8 ) const ; 
    std::string descBoundary() const ;
    std::string desc() const ; 

    unsigned getNumBoundary() const ; 
    const char* getBoundarySpec(unsigned idx) const ; 
    void        getBoundarySpec(std::vector<std::string>& names, const unsigned* idx , unsigned num_idx ) const ; 

    unsigned    getBoundaryIndex(const char* spec) const ;

    void        getBoundaryIndices( std::vector<unsigned>& bnd_idx, const char* bnd_sequence, char delim=',' ) const ; 
    std::string descBoundaryIndices( const std::vector<unsigned>& bnd_idx ) const ; 

    unsigned    getBoundaryLine(const char* spec, unsigned j) const ; 
    unsigned    getMaterialLine( const char* material ) const ; 

    static QTex<float4>* MakeBoundaryTex(const NP* buf ) ;
    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    NP*  lookup();
    void lookup( quad* lookup, unsigned num_lookup, unsigned width, unsigned height );
    void dump(   quad* lookup, unsigned num_lookup, unsigned edgeitems=10 );

};


