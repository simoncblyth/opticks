#pragma once

#include <vector>
#include "scuda.h"
#include "squad.h"
#include "plog/Severity.h"

struct SIMG ; 
#include "CSGOPTIX_API_EXPORT.hh"

struct CSGOPTIX_API Frame
{
    static const plog::Severity LEVEL ; 

    int width ; 
    int height ; 
    int depth ; 
    int channels ; 
    int jpg_quality ; 

    SIMG*      img ; 
    unsigned num_pixels ; 

    std::vector<float4> isect ; 
    std::vector<uchar4> pixel ; 
    std::vector<quad4>  photon ; 

    uchar4* d_pixel ; 
    float4* d_isect ; 
    quad4*  d_photon ; 

    Frame(int width_, int height_, int depth_, uchar4* d_pixel_=nullptr, float4* d_isect_=nullptr, quad4* d_photon_=nullptr ); 
 
    void download(); 

    void annotate( const char* bottom_line=nullptr, const char* top_line=nullptr, int line_height=24  );

    void write(const char* outdir, int quality=0) const ;

    void writePNG(const char* path) const ;
    void writePNG(const char* dir, const char* name) const ;

    void writeJPG(const char* path, int quality=0) const ;
    void writeJPG(const char* dir, const char* name, int quality=0) const ;

    void writeIsect(  const char* dir, const char* name) const ;
    void writePhoton( const char* dir, const char* name) const ;

    void snap(const char* path ); 


    unsigned getNumPixels() const ; 
    unsigned char* getPixelData() const ;
    float*         getIntersectData() const ;
    float*         getPhotonData() const ;

}; 

