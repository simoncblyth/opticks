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
    static const double FINE_STRUCTURE_OVER_HBARC_EVMM ; 


    static NP* Load(const char* path_) ; 

    const char*             path ; 
    NP*                     dsrc ; 
    double                  dmin ; 
    double                  dmax ; 

    NP*                     src ; 
    QTex<float>*            tex ; 

    QCerenkov(const char* path=nullptr); 

    void init(); 
    void makeTex(const NP* dsrc);
    std::string desc() const ; 

    template <typename T> T   getS2Integral( T& emin, T& emax, T& ecross, const T BetaInverse, const T en_0, const T en_1 , const T ri_0, const T ri_1, bool fix_cross ) const ; 

    template <typename T> NP* GetAverageNumberOfPhotons_s2_(T& emin,  T& emax, const T BetaInverse, const T  charge ) const ; 
    template <typename T> T   GetAverageNumberOfPhotons_s2(T& emin,  T& emax, const T BetaInverse, const T  charge ) const ; 

    template <typename T> NP* getS2CutIntegral_( const T BetaInverse, const T ecut ) const ; 
    template <typename T> T   getS2CutIntegral( const T BetaInverse, const T ecut ) const ; 

    template <typename T> NP* getS2CumulativeIntegrals( const T BetaInverse, unsigned nx ) const ; 
    template <typename T> NP* getS2CumulativeIntegrals( const NP* bis, unsigned nx ) const  ; 

    // TODO: remove the slivers as too approximate  
    template <typename T> NP* getS2SliverIntegrals( T& emin, T& emax, const T BetaInverse, const NP* edom ) const ; 
    template <typename T> NP* getS2SliverIntegrals( const NP* bis, const NP* edom ) const  ; 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void check();
    NP* lookup();
    void lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height ); 
    void dump(   float* lookup, unsigned num_lookup, unsigned edgeitems=10 ); 

};


