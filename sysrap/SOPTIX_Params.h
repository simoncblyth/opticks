#pragma once

#include <optix.h>


struct SOPTIX_Params
{ 
    unsigned width  ;
    unsigned height ;
    uchar4*  pixels  ;

    float  tmin ; 
    float  tmax ; 

    unsigned cameratype ; 
    float3 eye;
    float3 U;  
    float3 V;  
    float3 W;  

    OptixTraversableHandle handle ;

#ifndef __CUDACC__
    static SOPTIX_Params* d_param ;
    void device_alloc();
    void upload();  
#endif

};


#ifndef __CUDACC__

void SOPTIX_Params::device_alloc()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( SOPTIX_Params ) ) );   
    assert( d_param );  
}
void SOPTIX_Params::upload()
{
    assert( d_param );  
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( d_param ), this, sizeof( SOPTIX_Params ), cudaMemcpyHostToDevice) );  
}

#endif

