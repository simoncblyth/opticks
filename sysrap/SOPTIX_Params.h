#pragma once
/**
SOPTIX_Params.h : render control 
==================================
**/

#include <optix.h>


struct SOPTIX_Params
{ 
    unsigned width  ;
    unsigned height ;
    uchar4*  pixels  ;

    float  tmin ; 
    float  tmax ; 

    unsigned visibilityMask ; 
    unsigned cameratype ; 

    float3 eye;
    float3 U;  
    float3 V;  
    float3 W;  

    OptixTraversableHandle handle ;

#ifndef __CUDACC__
    SOPTIX_Params* device_alloc();
    void upload(SOPTIX_Params* d_param);  
#endif

};


#ifndef __CUDACC__

inline SOPTIX_Params* SOPTIX_Params::device_alloc()
{
    SOPTIX_Params* d_param = nullptr ;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( SOPTIX_Params ) ) );   
    return d_param ; 
}
inline void SOPTIX_Params::upload(SOPTIX_Params* d_param)
{
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_param ), this, sizeof( SOPTIX_Params ), cudaMemcpyHostToDevice) );  
    // hmm some optix7.5 SDK examples (optixMeshViewer) using cudaMemcpyAsync for parameter upload
}

#endif




