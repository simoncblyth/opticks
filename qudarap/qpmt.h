#pragma once
/**
qpmt.h
=======


**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QPMT_METHOD __device__
#else
   #define QPMT_METHOD 
#endif 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include "QUDARAP_API_EXPORT.hh"
#endif


template <typename T> struct qprop ;

template<typename T>
struct qpmt
{
    static constexpr const int NUM_CAT = 3 ; 
    static constexpr const int NUM_LAYR = 4 ; 
    static constexpr const int NUM_PROP = 2 ; 

    qprop<T>* rindex_prop ;
    qprop<T>* qeshape_prop ;

    T*        thickness ; 
    T*        lcqs ; 
}; 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template struct QUDARAP_API qpmt<float>;
template struct QUDARAP_API qpmt<double>;
#endif



