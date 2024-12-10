#pragma once

/**
QRng
====

Canonical instanciation within QSim::UploadComponents

Small *skipahead_event_offsets* are for functionality testing, 
typically the offset should be greater than the maximum number of 
randoms to simulate an item(photon). 



TODO : implement sanity check for use after loading::

    bool QRng::IsAllZero( RNG* states, unsigned num_states ) //  static

**/

#include <string>
#include "QUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"
#include "curand_kernel.h"   
#include "qrng.h"

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
    static std::string Desc();

    static const char* Load_FAIL_NOTES ; 
#ifdef OLD_MONOLITHIC_CURANDSTATE
    static constexpr const char* IMPL = "OLD_MONOLITHIC_CURANDSTATE" ; 

    static RNG* LoadAndUpload(ULL& rngmax, const char* path); 
    static RNG* Load(ULL& rngmax, const char* path); 
    static RNG* UploadAndFree(RNG* h_states, ULL num_states ); 
#else
    static constexpr const char* IMPL = "CHUNKED_CURANDSTATE" ; 
    static RNG* LoadAndUpload(ULL rngmax, const SCurandState& cs); 
    SCurandState   cs ; 
#endif
    static void Save( RNG* states, unsigned num_states, const char* path ); 

    const char*    path ; 
    ULL            rngmax ; 
    RNG*           d_rng_states ; 

    qrng*          qr ;  
    qrng*          d_qr ;  

    QRng(unsigned skipahead_event_offset=1) ;  
    void init(); 
    void initMeta(); 

    virtual ~QRng(); 

    void cleanup(); 
    std::string desc() const ; 


    template <typename T> void generate(      T* u, unsigned ni, unsigned nv, unsigned long long skipahead_ ) ; 
    template <typename T> void generate_evid( T* u, unsigned ni, unsigned nv, unsigned evid ) ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 

};

