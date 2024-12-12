#pragma once
/**
qrng.h
=======

TODO: incorporate the curand_init call from QCurandState.cu
into here and use this from there



**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QRNG_METHOD __device__
#else
   #define QRNG_METHOD 
#endif 

#include <curand_kernel.h>


#if defined(MOCK_CUDA)
#else

using XORWOW = curandStateXORWOW ;
using Philox = curandStatePhilox4_32_10 ; 

#ifdef WITH_CURANDLITE
using PhiloxLite = curandStatePhilox4_32_10_OpticksLite ; 
#endif

//using RNG = Philox ;
using RNG = XORWOW ;
#endif



struct qrng
{
    using ULL = unsigned long long ; 

    ULL  seed ;
    ULL  offset ; 
    ULL  skipahead_event_offset ; 

    void*   uploaded_states ; 


#if defined(__CUDACC__) || defined(__CUDABE__)
    QRNG_METHOD void get_rngstate_with_skipahead(RNG& rng, unsigned event_idx, unsigned photon_idx );  

#else
    qrng(ULL seed_, ULL offset_, ULL skipahead_event_offset_, void* uploaded_states_)
        :
        seed(seed_),
        offset(offset_),
        skipahead_event_offset(skipahead_event_offset_),
        uploaded_states(uploaded_states_)
    {
    }

#endif

}; 

#if defined(__CUDACC__) || defined(__CUDABE__)


/**
qrng::get_rngstate_with_skipahead  (formerly qrng::random_setup)
-----------------------------------------------------------------

light touch encapsulation of setup only as want generation of randoms to be familiar/standard and suffer no overheads

1. copy the *photon_idx* element of the rng_states array to the curandState reference argument

2. skipahead the curandState by skipahead_event_offset*event_idx 
   The offset can be configured using the OPTICKS_EVENT_SKIPAHEAD envvar.

   Ideally the offset value should be more than maximum number of random values consumed
   in any photon of any event. In practice the number of random values consumed per photon will 
   have a very long tail, so setting the event skipahead offset value to for example 10000 
   should prevent any clumping issues from the repeated use of the same randoms in every event.    


curand_init(seed, subsequence, offset, &rng) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

seed
   use as simulation constant

subsequence
   uses photon_idx

offset 
   use as simulation constant


With Philox the curand_init does skipahead and skipahead_sequence advancing ctr.xyzw::

    skipahead(offset,&rng)
       ctr.xyzw
       BUT: offset default zero 

    skipahead_sequence(subsequence,&rng)  
       ctr.zw


My current usage focusses on the "subsequence" for dimension

TODO: check performance implications of using different RNG "dimensions" 
for different purposes


**/

inline QRNG_METHOD void qrng::get_rngstate_with_skipahead(RNG& rng, unsigned event_idx, unsigned photon_idx )
{
    ULL skipahead_ = skipahead_event_offset*event_idx ; 
    ULL subsequence_ = photon_idx ; 

    if( uploaded_states == nullptr )
    {
        curand_init( seed, subsequence_, offset, &rng ) ;
    }
    else
    {
        rng = *( ((RNG*)uploaded_states) + photon_idx) ; 
    }    

    skipahead( skipahead_, &rng ); 
}
#endif
 
