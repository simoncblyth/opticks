#pragma once
#include "curand_kernel.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QCURAND_METHOD __device__
#else
   #define QCURAND_METHOD 
#endif 

template <typename T>
struct qcurand
{
#if defined(__CUDACC__) || defined(__CUDABE__)
   static QCURAND_METHOD T uniform( curandStateXORWOW* rng );  
#endif
};


#if defined(__CUDACC__) || defined(__CUDABE__)

template<> inline float qcurand<float>::uniform( curandStateXORWOW* rng ) 
{ 
#ifdef FLIP_RANDOM
    return 1.f - curand_uniform(rng) ;
#else
    return curand_uniform(rng) ;
#endif
}

template<> inline double qcurand<double>::uniform( curandStateXORWOW* rng ) 
{ 
#ifdef FLIP_RANDOM
    return 1. - curand_uniform_double(rng) ;
#else
    return curand_uniform_double(rng) ;
#endif
}

#endif


