#include <iostream>
#include <iomanip>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK


#include "NP.hh"
#include "Frame.h"
#include "Params.h"
#include "SPPM.h"

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"


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


void Frame::writePPM(const char* dir, const char* name, bool yflip ) const 
{
    std::cout << "Frame::writePPM " << dir << "/" << name << std::endl ; 
    SPPM_write( dir, name, pixels.data(), params->width, params->height, yflip );
}
void Frame::writePNG(const char* dir, const char* name) const 
{
    int channels = 4 ; 
    const unsigned char* data = (const unsigned char*)pixels.data();  
    SIMG img(int(params->width), int(params->height), channels,  data ); 
    img.writePNG(dir, name); 
}
void Frame::writeJPG(const char* dir, const char* name, int quality) const 
{
    int channels = 4 ; 
    const unsigned char* data = (const unsigned char*)pixels.data();  
    SIMG img(int(params->width), int(params->height), channels,  data ); 
    img.writeJPG(dir, name, quality); 
}

void Frame::writeNP( const char* dir, const char* name) const 
{
    std::cout << "Frame::writeNP " << dir << "/" << name << std::endl ; 
    NP::Write(dir, name, getIntersectData(), params->height, params->width, 4 );
}

float* Frame::getIntersectData() const
{
    return (float*)isect.data();
}



