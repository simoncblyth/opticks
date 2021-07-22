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

struct qprop
{
    const float* pp ; 
    unsigned width ; 
    unsigned height ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
    qprop( const float* pp, unsigned width, unsigned height ); 
    float interpolate( unsigned iprop, float x );  
#endif

}; 

#if defined(__CUDACC__) || defined(__CUDABE__)

inline QPROP_METHOD qprop::qprop( const float* pp_, unsigned width_, unsigned height_ )
    :
    pp(pp_), 
    width(width_),
    height(height_)
{
}

inline QPROP_METHOD float qprop::interpolate( unsigned iprop, float x )
{
    const float* vv = pp + width*iprop ; 

    union UIF
    {
       float f ; 
       int   i ; 
    }; 


    UIF u ; 
    u.f = *(vv+width-1) ; 
    int ni = u.i ; 

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

    float dy = vv[2*hi+1] - vv[2*lo+1] ; 
    float dx = vv[2*hi+0] - vv[2*lo+0] ; 
    float y = vv[2*lo+1] + dy*(x-vv[2*lo+0])/dx ; 
    return y ;  
}

#endif

