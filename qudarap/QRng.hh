#pragma once

/**
QRng
====

Aiming for a pure CUDA struct providing the essential features 
of optixrap/ORng but without any OptiX

Small *skipahead_event_offsets* are for functionality testing, 
typically the offset should be greater than the maximum number of 
randoms to simulate an item(photon). 

**/

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"
#include "curand_kernel.h"   // need header as curandState is typedef to curandXORWOW

struct qrng ; 


struct QUDARAP_API QRng 
{
    static const plog::Severity LEVEL ; 
    static const QRng* INSTANCE ; 
    static const char* DEFAULT_PATH ; 
    static const QRng* Get(); 

    static curandState* Load(long& rngmax, const char* path); 


    const char*    path ; 
    long           rngmax ; 
    curandState*   rng_states ; 
    qrng*          qr ;  
    qrng*          d_qr ;  

    QRng(const char* path=nullptr, unsigned skipahead_event_offset=1) ;  

    virtual ~QRng(); 

    void upload(); 
    void cleanup(); 
    std::string desc() const ; 


    template <typename T> void generate(   T* u, unsigned ni, unsigned nv, unsigned long long skipahead_ ) ; 
    template <typename T> void generate_2( T* u, unsigned ni, unsigned nv, unsigned event_idx ) ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 

};

