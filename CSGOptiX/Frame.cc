#include <iostream>
#include <iomanip>
#include <csignal>

#include <cuda_runtime.h>
#include "QU.hh"

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


uchar4* Frame::getDevicePixels() const { return d_pixels ;  }
float4* Frame::getDeviceIsect() const  { return d_isect ;  }


/**
Frame::init
-------------

Allocates pixels and isect on device. 

**/
void Frame::init()
{
    assert( depth == 1 ); 
    QU::device_free_and_alloc<uchar4>( d_pixels, width*height );  
    QU::device_free_and_alloc<float4>( d_isect, width*height );  
}

void Frame::download()
{
    download_pixels();  
    download_isect();  
}

void Frame::download_pixels()
{
    QU::Download<uchar4>(pixels, d_pixels, width*height ); 
    img->setData( getPixelsData() ); 
}

void Frame::download_isect()
{
    QU::Download<float4>(isect, d_isect, width*height ); 
}

unsigned char* Frame::getPixelsData() const { return (unsigned char*)pixels.data();  }
float*         Frame::getIntersectData() const { return (float*)isect.data(); }


void Frame::annotate( const char* bottom_line, const char* top_line, int line_height )
{
    img->annotate( bottom_line, top_line, line_height ); 
}

void Frame::write(const char* outdir, int jpg_quality) const 
{
    writePNG(outdir, "pixels.png");  
    writeJPG(outdir, "pixels.jpg", jpg_quality);  
    writeNP(  outdir, "posi.npy" );
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
    NP::Write(dir, name, getIntersectData(), height, width, 4 );
}

