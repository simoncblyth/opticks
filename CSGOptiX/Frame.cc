#include <iostream>
#include <iomanip>
#include <csignal>

#include <cuda_runtime.h>
#include "CUDA_CHECK.h"

#include "PLOG.hh"
#include "NP.hh"
#include "Frame.h"

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"


const plog::Severity Frame::LEVEL = PLOG::EnvLevel("Frame", "DEBUG" ); 


Frame::Frame(int width_, int height_, int depth_)
    :
    width(width_),
    height(height_),
    depth(depth_),
    channels(4),
    img(new SIMG(width, height, channels,  nullptr )) 
{
    init();
}

/**
Frame::init
-------------

Allocates pixels and isect on device. 

**/

void Frame::init()
{
    assert( depth == 1 ); 
    init_pixels(); 
    init_isect(); 
}

void Frame::init_pixels()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_pixels ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_pixels ),
                width*height*sizeof(uchar4)
                ) );
}

uchar4* Frame::getDevicePixels() const 
{
    return d_pixels ; 
}


void Frame::init_isect()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_isect ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_isect ),
                width*height*sizeof(float4)
                ) );
}
float4* Frame::getDeviceIsect() const 
{
    return d_isect ; 
}


void Frame::download()
{
    //LOG(LEVEL) ; 
    download_pixels();  
    download_isect();  
}

void Frame::download_pixels()
{
    //LOG(LEVEL) << "d_pixels " << d_pixels << std::endl ; 

    pixels.resize(width*height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( pixels.data() ),
                d_pixels,
                width*height*sizeof(uchar4),
                cudaMemcpyDeviceToHost
    ));

    img->setData( getPixelsData() ); 
}

unsigned char* Frame::getPixelsData() const
{
    unsigned char* data = (unsigned char*)pixels.data();  
    return data ; 
}

void Frame::download_isect()
{
    //LOG(LEVEL) << "d_isect " << d_isect << std::endl ; 

    isect.resize(width*height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( isect.data() ),
                d_isect,
                width*height*sizeof(float4),
                cudaMemcpyDeviceToHost
    ));
}


void Frame::annotate( const char* bottom_line, const char* top_line, int line_height )
{
    img->annotate( bottom_line, top_line, line_height ); 
}

void Frame::write(const char* outdir, int jpg_quality) const 
{
    //LOG(LEVEL) << outdir << std::endl ; 
    writePNG(outdir, "pixels.png");  
    writeJPG(outdir, "pixels.jpg", jpg_quality);  
    writeNP(  outdir, "posi.npy" );

    //std::raise(SIGINT); 
}

void Frame::writePNG(const char* dir, const char* name) const 
{
    img->writePNG(dir, name); 
}
void Frame::writePNG(const char* path) const 
{
    img->writePNG(path); 
}

void Frame::writeJPG(const char* dir, const char* name, int quality) const 
{
    img->writeJPG(dir, name, quality); 
}
void Frame::writeJPG(const char* path, int quality) const 
{
    img->writeJPG(path, quality); 
}

void Frame::writeNP( const char* dir, const char* name) const 
{
    //LOG(LEVEL) << dir << "/" << name ; 
    NP::Write(dir, name, getIntersectData(), height, width, 4 );
}
float* Frame::getIntersectData() const
{
    return (float*)isect.data();
}

