#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QPROP_METHOD __device__
#else
   #define QPROP_METHOD 
#endif 

/**
qprop
=======

**/

#include "sview.h"


template<typename T>
struct qprop
{
    T* pp ; 
    unsigned width ; 
    unsigned height ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
    QPROP_METHOD T  interpolate( unsigned iprop, T x );  
#else
    qprop()
        :
        pp(nullptr),
        width(0),
        height(0)
    {
    }
#endif

}; 





#if defined(__CUDACC__) || defined(__CUDABE__)

/**
qprop<T>::interpolate
-----------------------

1. access property data for index iprop
2. interpret the last column to obtain the number of payload values
3. binary search to find the bin relevant to domain argument x  
4. linear interpolation to yield the y value at x

**/

template <typename T>
inline QPROP_METHOD T qprop<T>::interpolate( unsigned iprop, T x )
{
    const T* vv = pp + width*iprop ; 

    int ni = sview::int_from<T>( *(vv+width-1) ) ; 

    int lo = 0 ;
    int hi = ni-1 ;

    if( x <= vv[2*lo+0] ) return vv[2*lo+1] ; 
    if( x >= vv[2*hi+0] ) return vv[2*hi+1] ; 

    while (lo < hi-1)
    {    
        int mi = (lo+hi)/2;
        if (x < vv[2*mi+0]) hi = mi ; 
        else lo = mi;
    }    

    T dy = vv[2*hi+1] - vv[2*lo+1] ; 
    T dx = vv[2*hi+0] - vv[2*lo+0] ; 
    T  y = vv[2*lo+1] + dy*(x-vv[2*lo+0])/dx ; 
    return y ;  
}


#endif

