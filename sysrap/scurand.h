#pragma once
/**
scurand.h
==========

When compiled with gcc (not nvcc) this uses srngcpu.h which mocks two methods
from the curand API enabling some CUDA code to be tested on CPU::

    curand_uniform
    curand_uniform_double

Users::

    opticks-fl scurand\<
    ./qudarap/qcerenkov_dev.h
    ./qudarap/QSim.cu
    ./qudarap/QRng.cu
    ./qudarap/qcerenkov.h
    ./sysrap/tests/scurand_test.cc
    ./sysrap/scurand.h


**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define SCURAND_METHOD __device__
   #include "curand_kernel.h"
#else
   #define SCURAND_METHOD
   #include "srngcpu.h"
#endif

#include "smath.h"



template <typename T>
struct scurand
{
   static SCURAND_METHOD T uniform( RNG* rng );
   static SCURAND_METHOD T shoot_exponential( RNG* rng );
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


template<typename T>
inline T scurand<T>::shoot_exponential( RNG* rng )
{
    T u = scurand<T>::uniform(rng);
    return -smath::log(u);
}





