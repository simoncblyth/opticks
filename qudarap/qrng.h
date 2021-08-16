#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QRNG_METHOD __device__
#else
   #define QRNG_METHOD 
#endif 


struct curandStateXORWOW ; 

struct qrng
{
    curandStateXORWOW*  rng_states ; 
    unsigned            skipahead_event_offset ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
    QRNG_METHOD void random_setup(curandStateXORWOW& rng, unsigned event_idx, unsigned photon_idx );  

#else
    qrng(unsigned skipahead_event_offset_)
        :
        rng_states(nullptr),
        skipahead_event_offset(skipahead_event_offset_)
    {
    }
#endif

}; 

#if defined(__CUDACC__) || defined(__CUDABE__)

#include <curand_kernel.h>

/**
qrng::random_setup
---------------------

light touch encapsulation of setup only as want generation of randoms to be familiar/standard and suffer no overheads

**/

inline QRNG_METHOD void qrng::random_setup(curandStateXORWOW& rng, unsigned event_idx, unsigned photon_idx )
{
    unsigned long long skipahead_ = skipahead_event_offset*event_idx ; 
    rng = *(rng_states + photon_idx) ; 
    skipahead( skipahead_, &rng ); 
}
#endif
 
