#include <iostream>
#include <iomanip>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

#include "Frame.h"
#include "Params.h"
#include "SPPM.h"
#include "NP.hh"


Frame::Frame(Params* params_)
    :
    params(params_)
{
    init();
}

void Frame::init()
{
    init_pixels(); 
    init_isect(); 
}


void Frame::init_pixels()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_pixels ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_pixels ),
                params->width*params->height*sizeof(uchar4)
                ) );
    params->pixels = d_pixels ; 
}

void Frame::init_isect()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_isect ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_isect ),
                params->width*params->height*sizeof(float4)
                ) );
    params->isect = d_isect ; 
}

void Frame::download()
{
    download_pixels();  
    download_isect();  
}

void Frame::download_pixels()
{
    pixels.resize(params->width*params->height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( pixels.data() ),
                d_pixels,
                params->width*params->height*sizeof(uchar4),
                cudaMemcpyDeviceToHost
    ));
}

void Frame::download_isect()
{
    isect.resize(params->width*params->height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( isect.data() ),
                d_isect,
                params->width*params->height*sizeof(float4),
                cudaMemcpyDeviceToHost
    ));
}


void Frame::writePPM(const char* path, bool yflip )
{
    std::cout << "Frame::writePPM " << path << std::endl ; 
    SPPM_write( path, pixels.data(), params->width, params->height, yflip );
}

void Frame::writeNPY(const char* path)
{
    int ni = params->height ; 
    int nj = params->width ;
    int nk = 4 ;  

    std::cout 
        << "Frame::writeNPY"
        << " ni  " << ni
        << " nj  " << nj
        << " nk  " << nk
        << " path " << path
        << std::endl 
        ;

    NP a("<f4", ni,nj,nk) ;    
    float* v = a.values<float>(); 
    float* isect_data = (float*)isect.data() ;

    for(int i=0 ; i < ni ; i++ ) 
    for(int j=0 ; j < nj ; j++ )
    for(int k=0 ; k < nk ; k++ )
    {
        int index =  i*nj*nk + j*nk + k ;
        *(v + index) = *(isect_data + index ) ;
    }
    a.save(path); 
}


