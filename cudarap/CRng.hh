#pragma once

/**
CRng
====

Aiming for a pure CUDA struct providing the essential features 
of optixrap/ORng but without any OptiX

**/

#ifdef __CUDACC__

__global__ void CRng_generate(int threads_per_launch, curandState* rng_states, float* d_arr )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;

    float u = curand_uniform(&rng_states[id]); 
    d_arr[id] = u ;   
}

#else

#include "CUDARAP_API_EXPORT.hh"
#include "plog/Severity.h"
#include "curand_kernel.h"

struct CUDARAP_API CRng 
{
    static const plog::Severity LEVEL ; 

    const char*   path ; 
    long          num_items ; 
    curandState*  d_rng_states ;    
    curandState*  rng_states ;    

    CRng(const char* path); 

    void init(); 
    int load(); 
    void upload(); 
};

#endif

