#pragma once

/**
QRng
====

Aiming for a pure CUDA struct providing the essential features 
of optixrap/ORng but without any OptiX

**/


#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"

#include "curand_kernel.h"   // needed because curandState is typedef to curandXORWOW


struct QUDARAP_API QRng 
{
    static const plog::Severity LEVEL ; 
    static const QRng* INSTANCE ; 
    static const char* DEFAULT_PATH ; 
    static const QRng* Get(); 

    const char*   path ; 
    long          rngmax ; 
    curandState*  d_rng_states ;    

    QRng(const char* path=nullptr); 
    virtual ~QRng(); 


    void load_and_upload(); 
    std::string desc() const ; 

    template <typename T>
    void generate( T* u, unsigned ni, unsigned nv, unsigned long long skipahead_ ) ; 


    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 

};

