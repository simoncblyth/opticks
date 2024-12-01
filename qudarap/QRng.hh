#pragma once

/**
QRng
====

Aiming for a pure CUDA struct providing the essential features 
of optixrap/ORng but without any OptiX

Small *skipahead_event_offsets* are for functionality testing, 
typically the offset should be greater than the maximum number of 
randoms to simulate an item(photon). 






TODO : implement sanity check for use after loading::

    bool QRng::IsAllZero( curandState* states, unsigned num_states ) //  static

**/

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"
#include "curand_kernel.h"   // need header as curandState is typedef to curandXORWOW


//#define OLD_MONOLITHIC_CURANDSTATE 1

#ifdef OLD_MONOLITHIC_CURANDSTATE
#else
#include "SCurandState.h"
#endif

struct qrng ; 


struct QUDARAP_API QRng 
{
    typedef unsigned long long ULL ; 
    static constexpr const char* init_VERBOSE = "QRng__init_VERBOSE" ; 
    static constexpr const ULL M = 1000000 ;  
    static const plog::Severity LEVEL ; 
    static const QRng* INSTANCE ; 
    static const QRng* Get(); 

    static const char* Load_FAIL_NOTES ; 
#ifdef OLD_MONOLITHIC_CURANDSTATE
    static curandState* LoadAndUpload(ULL& rngmax, const char* path); 
    static curandState* Load(ULL& rngmax, const char* path); 
    static curandState* UploadAndFree(curandState* h_states, ULL num_states ); 
#else
    static curandState* LoadAndUpload(ULL rngmax, const _SCurandState& cs); 
    _SCurandState   cs ; 
#endif
    static void Save( curandState* states, unsigned num_states, const char* path ); 

    const char*    path ; 
    ULL            rngmax ; 
    curandState*   d_rng_states ; 

    qrng*          qr ;  
    qrng*          d_qr ;  

    QRng(unsigned skipahead_event_offset=1) ;  
    void init(); 
    void initMeta(); 

    virtual ~QRng(); 

    void cleanup(); 
    std::string desc() const ; 


    template <typename T> void generate(   T* u, unsigned ni, unsigned nv, unsigned long long skipahead_ ) ; 
    template <typename T> void generate_2( T* u, unsigned ni, unsigned nv, unsigned event_idx ) ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 

};

