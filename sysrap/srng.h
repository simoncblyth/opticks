#pragma once
/**
srng.h : picks curandState implementation 
===========================================

This is included into qudarap/qrng.h 

Template specializations collecting details of the various curandState impls.  See::

    ~/o/sysrap/tests/srng_test.sh 


https://stackoverflow.com/questions/8789867/c-template-class-specialization-why-do-common-methods-need-to-be-re-implement

Each specialisation of a class template gives a different class - they do not share any members.
So have to implement all methods in each specialization, or use a separate helper. 

**/

#include <curand_kernel.h>

using XORWOW = curandStateXORWOW ;
using Philox = curandStatePhilox4_32_10 ; 

#if defined(RNG_PHILITEOX)
#include "curandlite/curandStatePhilox4_32_10_OpticksLite.h"
using PhiLiteOx = curandStatePhilox4_32_10_OpticksLite ; 
#endif

#if defined(RNG_XORWOW)
using RNG = XORWOW ;
#elif defined(RNG_PHILOX)
using RNG = Philox ;
#elif defined(RNG_PHILITEOX)
using RNG = PhiLiteOx ;
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <sstream>
#include <string>

template<typename T> struct srng {};

// template specializations for the different states
template<> 
struct srng<XORWOW>  
{ 
    static constexpr char CODE = 'X' ;
    static constexpr const char* NAME = "XORWOW" ; 
    static constexpr unsigned SIZE = sizeof(XORWOW) ; 
    static constexpr bool UPLOAD_RNG_STATES = true ; 
};

template<> 
struct srng<Philox>  
{ 
    static constexpr char CODE = 'P' ;
    static constexpr const char* NAME = "Philox" ; 
    static constexpr unsigned SIZE = sizeof(Philox) ; 
    static constexpr bool UPLOAD_RNG_STATES = false ; 
};

#if defined(RNG_PHILITEOX)
template<> 
struct srng<PhiLiteOx>  
{ 
    static constexpr char CODE = 'O' ;
    static constexpr const char* NAME = "PhiLiteOx" ; 
    static constexpr unsigned SIZE = sizeof(PhiLiteOx) ; 
    static constexpr bool UPLOAD_RNG_STATES = false ; 
};
#endif

// helper function
template<typename T> 
inline std::string srng_Desc()
{
    std::stringstream ss ; 
    ss 
       << "[srng_Desc\n" 
       <<  " srng<T>::NAME " << srng<T>::NAME << "\n"
       <<  " srng<T>::CODE " << srng<T>::CODE << "\n"
       <<  " srng<T>::SIZE " << srng<T>::SIZE << "\n"
       << "]srng_Desc" 
       ; 
    std::string str = ss.str() ; 
    return str ; 
}

template<typename T> 
inline bool srng_IsXORWOW(){ return strcmp(srng<T>::NAME, "XORWOW") == 0 ; }

template<typename T> 
inline bool srng_IsPhilox(){ return strcmp(srng<T>::NAME, "Philox") == 0 ; }

template<typename T> 
inline bool srng_IsPhiLiteOx(){ return strcmp(srng<T>::NAME, "PhiLiteOx") == 0 ; }

template<typename T> 
inline bool srng_Matches(const char* arg)
{
    int match = 0 ; 
    if( arg && strstr(arg, "XORWOW")    && srng_IsXORWOW<T>() )    match += 1 ; 
    if( arg && strstr(arg, "Philox")    && srng_IsPhilox<T>() )    match += 1 ; 
    if( arg && strstr(arg, "PhiLiteOx") && srng_IsPhiLiteOx<T>() ) match += 1 ; 
    return match == 1 ; 
} 

#endif

