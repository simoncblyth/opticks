#pragma once
/**
srng.h
========

Template specializations collecting details of the 
various curandState impls.  See::

    ~/o/sysrap/tests/srng_test.sh 


https://stackoverflow.com/questions/8789867/c-template-class-specialization-why-do-common-methods-need-to-be-re-implement

Each specialisation of a class template gives a different class - they do not share any members.
So have to implement all methods in each specialization, or use a separate helper. 

**/

#include <curand_kernel.h>
#include <sstream>
#include <string>

template<typename T> 
struct srng
{ 
    static constexpr const char CODE = '?' ;
    static constexpr const char* NAME = "?" ;
    static constexpr unsigned SIZE = 0 ;
};

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


template<> 
struct srng<curandStateXORWOW>  
{ 
    static constexpr char CODE = 'X' ;
    static constexpr const char* NAME = "XORWOW" ; 
    static constexpr unsigned SIZE = sizeof(curandStateXORWOW) ; 
};

template<> 
struct srng<curandStatePhilox4_32_10>  
{ 
    static constexpr char CODE = 'P' ;
    static constexpr const char* NAME = "Philox4_32_10" ; 
    static constexpr unsigned SIZE = sizeof(curandStatePhilox4_32_10) ; 
};



#ifdef WITH_CURANDLITE
#include "curandlite/curandStatePhilox4_32_10_OpticksLite.h"

template<> 
struct srng<curandStatePhilox4_32_10_OpticksLite>  
{ 
    static constexpr char CODE = 'O' ;
    static constexpr const char* NAME = "Philox4_32_10_OpticksLite" ; 
    static constexpr unsigned SIZE = sizeof(curandStatePhilox4_32_10_OpticksLite) ; 
};

#endif

