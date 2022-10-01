#include <iostream>
#include <iomanip>
#include <csignal>

#include <cuda_runtime.h>
#include "QU.hh"

#include "SFrameConfig.hh"
#include "SComp.h"
#include "SStr.hh"
#include "SPath.hh"
#include "SLOG.hh"
#include "NP.hh"
#include "Frame.h"

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"


const plog::Severity Frame::LEVEL = SLOG::EnvLevel("Frame", "DEBUG" ); 

unsigned Frame::getNumPixels() const 
{
    return num_pixels ; 
}

/**
Frame::Frame
--------------

Instanciated by:

1. CSGOptiX::CSGOptiX with null device pointer args
2. Six::Six with device pointers fed in 

Accepting device buffer pointer arguments was done to allow this 
class to be used with OptiX 6 workflow optix::Buffer

HMM: could use QEvent to hold the pixel, isect, photon ?
**/


template<typename T> 
T* Frame::DeviceAlloc(unsigned num_pixels, bool enabled)
{
    return enabled ? QU::device_alloc<T>(num_pixels,"Frame::DeviceAllo:num_pixels") : nullptr ; 
}

template uchar4* Frame::DeviceAlloc<uchar4>(unsigned num_pixels, bool enabled); 
template float4* Frame::DeviceAlloc<float4>(unsigned num_pixels, bool enabled); 
template quad4*  Frame::DeviceAlloc<quad4>( unsigned num_pixels, bool enabled); 


Frame::Frame(int width_, int height_, int depth_, uchar4* d_pixel_, float4* d_isect_, quad4* d_fphoton_ )
    :
    mask(SFrameConfig::FrameMask()),
    width(width_),
    height(height_),
    depth(depth_),
    channels(4),
    jpg_quality(SStr::GetEValue<int>("QUALITY", 50)),
    img(new SIMG(width, height, channels,  nullptr )),
    num_pixels(width*height),  
    d_pixel(d_pixel_ == nullptr     ? DeviceAlloc<uchar4>(num_pixels, mask & SCOMP_PIXEL   ) : d_pixel_  ),
    d_isect(d_isect_ == nullptr     ? DeviceAlloc<float4>(num_pixels, mask & SCOMP_ISECT   ) : d_isect_  ),
#ifdef WITH_FRAME_PHOTON
    d_fphoton(d_fphoton_ == nullptr ? DeviceAlloc<quad4>( num_pixels, mask & SCOMP_FPHOTON ) : d_fphoton_)
#else
    d_dummy(nullptr)  
#endif
{
    assert( depth == 1 && num_pixels > 0 ); 
}

/**
Frame::download from GPU buffers into vectors
-----------------------------------------------

This is invoked from CSGOptiX::snap

**/
void Frame::download()
{
    if(d_pixel)   QU::Download<uchar4>(pixel, d_pixel, num_pixels ); 
    if(d_isect)   QU::Download<float4>(isect, d_isect, num_pixels ); 
#ifdef WITH_FRAME_PHOTON
    if(d_fphoton) QU::Download<quad4>(photon, d_fphoton, num_pixels ); 
#endif
    if(d_pixel) img->setData( getPixelData() ); 
}



unsigned char* Frame::getPixelData() const {     return d_pixel ? (unsigned char*)pixel.data() : nullptr ; }
float*         Frame::getIntersectData() const { return d_isect ? (float*)isect.data()         : nullptr ; }
#ifdef WITH_FRAME_PHOTON
float*         Frame::getFPhotonData() const {   return d_fphoton ? (float*)fphoton.data()     : nullptr ; }
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
    writeFPhoton(outdir, "f_photon.npy" ); 
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
    float* isd = getIntersectData() ;
    if(isd) NP::Write(dir, name, isd, height, width, 4 );
}

#ifdef WITH_FRAME_PHOTON
void Frame::writeFPhoton( const char* dir, const char* name) const 
{
    float* fpd = getFPhotonData() ;  
    if(fpd) NP::Write(dir, name, fpd, height, width, 4, 4 );
}
#endif

void Frame::snap( const char* path )
{
    LOG(LEVEL) << "[" ; 

    LOG(LEVEL) << "[ writeJPG " ; 
    writeJPG( path ); 
    LOG(LEVEL) << "] writeJPG " ; 


    LOG(LEVEL) << "[ writeIntersectData " ; 
    const char* fold = SPath::Dirname(path); 
    float* isd = getIntersectData() ;
    if(isd) NP::Write(fold, "isect.npy", isd, height, width, 4 );
    LOG(LEVEL) << "] writeIntersectData " ; 


#ifdef WITH_FRAME_PHOTON
    LOG(LEVEL) << "[ writeFPhoton " ; 
    float* fpd = getFPhotonData() ;  
    if(fpd) NP::Write(fold, "fphoton.npy", fpd, height, width, 4, 4 );
    LOG(LEVEL) << "] writeFPhoton " ; 
#endif

    LOG(LEVEL) << "]" ; 
}


