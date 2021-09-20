#pragma once

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct NP ; 
template <typename T> struct QTex ; 
template <typename T> struct QCK ; 
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

    enum { NONE, UNCUT, CUT, FULL, PART, ERR };  

    static const char* NONE_ ; 
    static const char* UNCUT_ ; 
    static const char* CUT_ ; 
    static const char* FULL_ ; 
    static const char* PART_ ; 
    static const char* ERR_ ; 
    static const char* State(int state); 

    static NP* Load(const char* path_) ; 

    const char*             path ; 
    NP*                     dsrc ;  // RINDEX array 

    double                  emn ; 
    double                  emx ; 
    double                  rmn ; 
    double                  rmx ; 

    NP*                     src ; 
    QTex<float>*            tex ; 

    QCerenkov(const char* path=nullptr); 

    void init(); 
    void makeTex(const NP* dsrc);
    std::string desc() const ; 

    // fixed bin subdiv, wastes bins but its simpler and should avoid slight non-monotonic issue
    template<typename T> unsigned getS2Edges(unsigned mul ) const ;
    template <typename T> NP* getS2Integral_Cumulative( const NP* bis, unsigned mul ) const ; 
    template <typename T> NP* getS2Integral_Cumulative(const T BetaInverse, unsigned mul ) const ; 


    template <typename T> T   getS2Integral_WithCut(  T& emin, T& emax, const T BetaInverse, const T en_cut, bool dump ) const  ; 
    template <typename T> NP* getS2Integral_WithCut_( T& emin, T& emax, const T BetaInverse, const T en_cut, bool dump  ) const  ; 
    template <typename T> static T GetS2Integral_WithCut( T& emin, T& emax, const T BetaInverse, const T en_0, const T en_1 , const T ri_0, const T ri_1, const T en_cut, const T ri_cut, bool dump ) ; 

    template <typename T> T getS2( const T BetaInverse, const T en ) const ; 

    template <typename T> NP* GetAverageNumberOfPhotons_s2_(T& emin,  T& emax, const T BetaInverse, const T  charge ) const ; 
    template <typename T> T   GetAverageNumberOfPhotons_s2(T& emin,  T& emax, const T BetaInverse, const T  charge ) const ; 

    // hmm "make" rather than "get" 
    template <typename T> NP* getS2CumulativeIntegrals( const T BetaInverse, unsigned nx ) const ; 
    template <typename T> NP* getS2CumulativeIntegrals( const NP* bis, unsigned nx ) const  ; 

    template <typename T> QCK<T> makeICDF( unsigned ny, unsigned nx ) const ; 


    // TODO: remove the slivers as too approximate  
    //template <typename T> NP* getS2SliverIntegrals( T& emin, T& emax, const T BetaInverse, const NP* edom ) const ; 
    //template <typename T> NP* getS2SliverIntegrals( const NP* bis, const NP* edom ) const  ; 

    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );

    void check();
    NP* lookup();
    void lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height ); 
    void dump(   float* lookup, unsigned num_lookup, unsigned edgeitems=10 ); 

};


