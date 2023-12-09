#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define LEAF_FUNC __forceinline__ __device__
#else
#    define LEAF_FUNC inline
#endif


#define RT_DEFAULT_MAX 1.e27f

#if defined(__CUDACC__)
#include "math_constants.h"
#else

union uif_t 
{
    unsigned u ; 
    int i ; 
    float f ; 
};

LEAF_FUNC
float __int_as_float(int i)
{
    uif_t uif ; 
    uif.i = i ; 
    return uif.f ; 
}

#define CUDART_INF_F            __int_as_float(0x7f800000)
#define CUDART_PI_F             3.141592654f

#endif



