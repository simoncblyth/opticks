#pragma once
/**
qrng.h
=======

Despite differences between template specializations regarding *uploaded_states* 
all specializations have the same init_with_skipahead signature. 
The ctor is only compiled on CPU, as the instance get instanciated on CPU 
before being uploaded to GPU. 


init_with_skipahead
---------------------

1. With XORWOW copy the *photon_idx* element of uploaded_states array 
   to the curandState reference argument, with others curand_init is cheap, 
   so is used directly.  For XORWOW the curand_init is done in separate
   launches creating the chunked states files with QCurandState.cu

   * hence only qrng<XORWOW> template specialization has *uploaded_states* member

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
   [current usage focusses on this RNG "dimension"]

   * TODO: check implications of using different RNG "dimensions" for different purposes


offset 
   use as simulation constant


**/

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QRNG_METHOD __device__
#else
   #define QRNG_METHOD 
#endif 

#include "srng.h"

using ULL = unsigned long long ; 

template<typename T> struct qrng {} ; 

template<> 
struct qrng<XORWOW>
{
    ULL  seed ;
    ULL  offset ; 
    ULL  skipahead_event_offset ; 


#if defined(__CUDACC__) || defined(__CUDABE__)
    XORWOW*   uploaded_states ; 
#else
    void*     uploaded_states ; 
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
    QRNG_METHOD void init(XORWOW& rng, unsigned event_idx, unsigned photon_idx )
    {  
        rng = uploaded_states[photon_idx] ; 
        ULL skipahead_ = skipahead_event_offset*event_idx ; 
        skipahead( skipahead_, &rng ); 
    }
#else
    qrng(ULL seed_, ULL offset_, ULL skipahead_event_offset_ )
        :
        seed(seed_),
        offset(offset_),
        skipahead_event_offset(skipahead_event_offset_),
        uploaded_states(nullptr)
    {
    }

    void set_uploaded_states( void*  uploaded_states_ )
    {
        uploaded_states = uploaded_states_ ;
    } 

#endif
}; 


/**
qrng<Philox>
-------------

With Philox the curand_init does skipahead and skipahead_sequence advancing ctr.xyzw:

skipahead(offset,&rng)
   ctr.xyzw

skipahead_sequence(subsequence,&rng)  
   ctr.zw

**/

template<> 
struct qrng<Philox>
{
    ULL  seed ;
    ULL  offset ; 
    ULL  skipahead_event_offset ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
    QRNG_METHOD void init(Philox& rng, unsigned event_idx, unsigned photon_idx )
    {  
        ULL subsequence_ = photon_idx ; 
        curand_init( seed, subsequence_, offset, &rng ) ;
        ULL skipahead_ = skipahead_event_offset*event_idx ; 
        skipahead( skipahead_, &rng ); 
    }
#else
    qrng(ULL seed_, ULL offset_, ULL skipahead_event_offset_ )
        :
        seed(seed_),
        offset(offset_),
        skipahead_event_offset(skipahead_event_offset_)
    {
    }
    void set_uploaded_states( void* ){}
#endif
}; 


#ifdef RNG_PHILITEOX
template<> 
struct qrng<PhiloxLite>
{
    ULL  seed ;
    ULL  offset ; 
    ULL  skipahead_event_offset ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
    QRNG_METHOD void init(PhiloxLite& rng, unsigned event_idx, unsigned photon_idx )
    {  
        ULL subsequence_ = photon_idx ; 
        curand_init( seed, subsequence_, offset, &rng ) ;
        ULL skipahead_ = skipahead_event_offset*event_idx ; 
        skipahead( skipahead_, &rng ); 
    }
#else
    qrng(ULL seed_, ULL offset_, ULL skipahead_event_offset_ )
        :
        seed(seed_),
        offset(offset_),
        skipahead_event_offset(skipahead_event_offset_)
    {
    }
    void set_uploaded_states( void* ){}
#endif
}; 
#endif


