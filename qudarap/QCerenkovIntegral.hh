#pragma once

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

struct float4 ; 
struct NP ; 
template <typename T> struct QCK ; 


/**
QCerenkovIntegral : creates set of ICDF for many different BetaInverse values
================================================================================

TODO : this integration is not using CUDA at all...  so  this and QCK needs to be moved elsewhere, maybe SCerenkovIntegral, SCK ?
That will enable Cerenkov ICDF creation pre-cache in CSG_GGeo

Prototyping/experimentation done in ana/rindex.py 
**/

struct QUDARAP_API QCerenkovIntegral
{
    static const plog::Severity LEVEL ; 
    static const char* DEFAULT_PATH ; 
    static const double FINE_STRUCTURE_OVER_HBARC_EVMM ; 

    static const unsigned UPPERCUT_PAYLOAD_SIZE ; 
    static const unsigned SPLITBIN_PAYLOAD_SIZE ; 

    enum { NONE, UNCUT, CUT, SUB, FULL, PART, ERR };  

    static const char* NONE_ ; 
    static const char* UNCUT_ ; 
    static const char* CUT_ ; 
    static const char* SUB_ ; 
    static const char* FULL_ ; 
    static const char* PART_ ; 
    static const char* ERR_ ; 

    static const char* State(int state); 


    const char*             path ; 
    NP*                     dsrc ;  // RINDEX array 
    NP*                     src ; 

    double                  emn ; 
    double                  emx ; 
    double                  rmn ; 
    double                  rmx ; 


    QCerenkovIntegral(const char* rindex_path=nullptr); 
    void init(); 


    std::string desc() const ; 
    template <typename T> NP* GetAverageNumberOfPhotons_s2_(T& emin,  T& emax, const T BetaInverse, const T  charge ) const ; 
    template <typename T> T   GetAverageNumberOfPhotons_s2(T& emin,  T& emax, const T BetaInverse, const T  charge ) const ; 
    template <typename T> NP* getAverageNumberOfPhotons_s2( const NP* bis ) const ; 

    // fixed bin subdiv, wastes bins but its simpler and should avoid slight non-monotonic issue
    template<typename T> unsigned getNumEdges_SplitBin(unsigned mul ) const ;
    template <typename T> NP* getS2Integral_SplitBin( const NP* bis, unsigned mul, bool dump ) const ; 
    template <typename T> NP* getS2Integral_SplitBin(const T BetaInverse, unsigned mul, bool dump ) const ; 
    template <typename T> QCK<T> makeICDF_SplitBin( unsigned ny, unsigned mul, bool dump ) const ; 


    template <typename T> T   getS2Integral_WithCut(  T& emin, T& emax, T BetaInverse, T en_a, T en_b, bool dump ) const  ; 
    template <typename T> NP* getS2Integral_WithCut_( T& emin, T& emax, T BetaInverse, T en_a, T en_b, bool dump ) const  ; 
    template <typename T> static T GetS2Integral_WithCut( T& emin, T& emax,  T BetaInverse,  T en_0,  T en_1 ,  T ri_0,  T ri_1, T en_a, T ri_a, T en_b, T ri_b, bool dump ) ; 

    template <typename T> T getS2( const T BetaInverse, const T en ) const ; 

    template <typename T> NP* getS2Integral_UpperCut( const T BetaInverse, unsigned nx ) const ; 
    template <typename T> NP* getS2Integral_UpperCut( const NP* bis, unsigned nx ) const  ; 
    template <typename T> QCK<T> makeICDF_UpperCut( unsigned ny, unsigned nx, bool dump ) const ; 

};


