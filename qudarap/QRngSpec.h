#pragma once
/**
QRngSpec.h
===========

Template specializations collecting details of the 
various curandState impls.  See::

    ~/o/qudarap/tests/QRngSpec_test.sh 

**/

#include <curand_kernel.h>
#include <sstream>
#include <string>

template<typename T> 
struct QRngSpec
{ 
    static constexpr const char CODE = '?' ;
    static constexpr const char* NAME = "?" ;
    static constexpr unsigned SIZE = 0 ;
   // static std::string Desc(); 
};

/*
// template specializations dont allow this it seems
template<typename T> 
inline std::string QRngSpec<T>::Desc()
{
    std::stringstream ss ; 
    ss 
       << "[QRngSpec::Desc" 
       <<  " NAME " << NAME << "\n"
       <<  " CODE " << CODE << "\n"
       <<  " SIZE " << SIZE << "\n"
       << "]QRngSpec::Desc" 
       ; 
    std::string str = ss.str() ; 
    return str ; 
}
*/


template<> 
struct QRngSpec<curandStateXORWOW>  
{ 
    static constexpr char CODE = 'X' ;
    static constexpr const char* NAME = "XORWOW" ; 
    static constexpr unsigned SIZE = sizeof(curandStateXORWOW) ; 
};

template<> 
struct QRngSpec<curandStatePhilox4_32_10>  
{ 
    static constexpr char CODE = 'P' ;
    static constexpr const char* NAME = "Philox4_32_10" ; 
    static constexpr unsigned SIZE = sizeof(curandStatePhilox4_32_10) ; 
};



#ifdef WITH_CURANDLITE
#include "curandlite/curandStatePhilox4_32_10_OpticksLite.h"

template<> 
struct QRngSpec<curandStatePhilox4_32_10_OpticksLite>  
{ 
    static constexpr char CODE = 'O' ;
    static constexpr const char* NAME = "Philox4_32_10_OpticksLite" ; 
    static constexpr unsigned SIZE = sizeof(curandStatePhilox4_32_10_OpticksLite) ; 
};

#endif

