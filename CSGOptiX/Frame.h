#pragma once
/**
Frame.h : Frame as in window, not volume 
===========================================

When ctor argument pointers are not provided the ctor allocates device buffers.
This was done for compatibility with the pre-OptiX-7 API.


**/

#include <vector>
#include "scuda.h"
#include "squad.h"
#include "plog/Severity.h"

struct SIMG ; 
#include "CSGOPTIX_API_EXPORT.hh"

struct CSGOPTIX_API Frame
{
    static const plog::Severity LEVEL ; 

    unsigned mask ; 
    int width ; 
    int height ; 
    int depth ; 
    int channels ; 
    int jpg_quality ; 

    SIMG*      img ; 
    unsigned num_pixels ; 

    std::vector<float4> isect ; 
    std::vector<uchar4> pixel ; 
#ifdef WITH_FRAME_PHOTON
    std::vector<quad4>  fphoton ; 
#endif

    template<typename T> 
    static T* DeviceAlloc(unsigned num_pixels, bool enabled ); 

    uchar4* d_pixel ; 
    float4* d_isect ; 
#ifdef WITH_FRAME_PHOTON
    quad4*  d_fphoton ; 
#else
    quad4*  d_dummy ; 
#endif

public:
    Frame(int width_, int height_, int depth_, uchar4* d_pixel_=nullptr, float4* d_isect_=nullptr, quad4* d_fphoton_=nullptr ); 
    void setExternalDevicePixels(uchar4* _d_pixel );
    void download(); 
    void annotate( const char* bottom_line=nullptr, const char* top_line=nullptr, int line_height=24  );
    void snap(const char* path ); 

private: 

    void write(const char* outdir, int quality=0) const ;

    void writePNG(const char* path) const ;
    void writePNG(const char* dir, const char* name) const ;

    void writeJPG(const char* path, int quality=0) const ;
    void writeJPG(const char* dir, const char* name, int quality=0) const ;

    void writeIsect(  const char* dir, const char* name) const ;


    unsigned getNumPixels() const ; 
    unsigned char* getPixelData() const ;
    float*         getIntersectData() const ;

#ifdef WITH_FRAME_PHOTON
    void writeFPhoton( const char* dir, const char* name) const ;
    float*         getFPhotonData() const ;
#endif

}; 

