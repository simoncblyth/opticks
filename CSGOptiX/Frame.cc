#include <iostream>
#include <iomanip>
#include <csignal>

#include <cuda_runtime.h>
#include "QU.hh"

#include "SStr.hh"
#include "SPath.hh"
#include "PLOG.hh"
#include "NP.hh"
#include "Frame.h"

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"


const plog::Severity Frame::LEVEL = PLOG::EnvLevel("Frame", "DEBUG" ); 

unsigned Frame::getNumPixels() const 
{
    return num_pixels ; 
}

/**

HMM: could use QEvent to hold the pixel, isect, photon ?

**/


Frame::Frame(int width_, int height_, int depth_, uchar4* d_pixel_, float4* d_isect_, quad4* d_photon_ )
    :
    width(width_),
    height(height_),
    depth(depth_),
    channels(4),
    jpg_quality(SStr::GetEValue<int>("QUALITY", 50)),
    img(new SIMG(width, height, channels,  nullptr )),
    num_pixels(width*height),  
    d_pixel(d_pixel_ == nullptr ? QU::device_alloc<uchar4>(num_pixels) : d_pixel_),
    d_isect(d_isect_ == nullptr ? QU::device_alloc<float4>(num_pixels) : d_isect_),
#ifdef WITH_FRAME_PHOTON
    d_photon(d_photon_ == nullptr ? QU::device_alloc<quad4>(num_pixels) : d_photon_)
#else
    d_dummy(nullptr)  
#endif
{
    assert( depth == 1 && num_pixels > 0 ); 
}

/**
Frame::download
----------------

Download from GPU buffers into vectors.

**/
void Frame::download()
{
    QU::Download<uchar4>(pixel, d_pixel, num_pixels ); 
    QU::Download<float4>(isect, d_isect, num_pixels ); 
#ifdef WITH_FRAME_PHOTON
    QU::Download<quad4>(photon, d_photon, num_pixels ); 
#endif

    img->setData( getPixelData() ); 
}

unsigned char* Frame::getPixelData() const {     return (unsigned char*)pixel.data();  }
float*         Frame::getIntersectData() const { return (float*)isect.data(); }
#ifdef WITH_FRAME_PHOTON
float*         Frame::getPhotonData() const {    return (float*)photon.data(); }
#endif


void Frame::annotate( const char* bottom_line, const char* top_line, int line_height )
{
    img->annotate( bottom_line, top_line, line_height ); 
}

void Frame::write(const char* outdir_, int quality) const 
{
    const char* outdir = SPath::Resolve(outdir_, DIRPATH); 
    writePNG(outdir, "f_pixels.png");  
    writeJPG(outdir, "f_pixels.jpg", quality);  
    writeIsect(outdir, "f_isect.npy" ); // formerly posi.npy
#ifdef WITH_FRAME_PHOTON
    writePhoton(outdir, "f_photon.npy" ); 
#endif
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
    img->writeJPG(dir, name, quality > 0 ? quality : jpg_quality ); 
}
void Frame::writeJPG(const char* path, int quality) const 
{
    img->writeJPG(path, quality > 0 ? quality : jpg_quality ); 
}


void Frame::writeIsect( const char* dir, const char* name) const 
{
    NP::Write(dir, name, getIntersectData(), height, width, 4 );
}
#ifdef WITH_FRAME_PHOTON
void Frame::writePhoton( const char* dir, const char* name) const 
{
    NP::Write(dir, name, getPhotonData(), height, width, 4, 4 );
}
#endif

void Frame::snap( const char* path )
{
    writeJPG( path ); 
    const char* fold = SPath::Dirname(path); 
    NP::Write(fold, "isect.npy", getIntersectData(), height, width, 4 );
#ifdef WITH_FRAME_PHOTON
    NP::Write(fold, "photon.npy", getPhotonData(), height, width, 4, 4 );
#endif

}


