#pragma once

// http://stackoverflow.com/questions/12778949/cuda-memory-alignment

#if defined(__CUDACC__) // NVCC
    #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
    #define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
    #define MY_ALIGN(n) __declspec(align(n))
#elif defined(__clang__) // 
    #define MY_ALIGN(n) __attribute__((aligned(n)))
#else
    #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif


#include "quad.h"
typedef struct MY_ALIGN(16) { float4 a,b,c,d ; } photon_t ;

photon_t make_photon()
{
    photon_t p ; 
    p.a = make_float4( 0.f, 0.f, 0.f, 0.f );
    p.b = make_float4( 1.f, 1.f, 1.f, 1.f );
    p.c = make_float4( 2.f, 2.f, 2.f, 2.f );
    p.d = make_float4( 3.f, 3.f, 3.f, 3.f );
    return p ; 
}

photon_t make_photon_tagged(unsigned int code)
{
    photon_t p ; 
    p.a = make_float4( 0.f, 0.f, 0.f, 0.f );
    p.b = make_float4( 1.f, 1.f, 1.f, 1.f );
    p.c = make_float4( 2.f, 2.f, 2.f, 2.f );

    quad q ; 
    q.u = make_uint4( 0u, 0u, 0u, code );
    p.d = q.f ;

    return p ; 
}




