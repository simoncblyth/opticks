#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define SCURAND_METHOD __device__
   #include "curand_kernel.h"
#else
   #define SCURAND_METHOD 
   #include "srngcpu.h"
#endif 

template <typename T>
struct scurand
{
   static SCURAND_METHOD T uniform( RNG* rng );  
};



template<> inline float scurand<float>::uniform( RNG* rng ) 
{ 
#ifdef FLIP_RANDOM
    return 1.f - curand_uniform(rng) ;
#else
    return curand_uniform(rng) ;
#endif
}

template<> inline double scurand<double>::uniform( RNG* rng ) 
{ 
#ifdef FLIP_RANDOM
    return 1. - curand_uniform_double(rng) ;
#else
    return curand_uniform_double(rng) ;
#endif
}



