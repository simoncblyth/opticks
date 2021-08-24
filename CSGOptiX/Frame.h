#pragma once

#include <vector>
#include "scuda.h"
#include "squad.h"
#include "plog/Severity.h"

struct SIMG ; 

struct Frame
{
    static const plog::Severity LEVEL ; 

    int width ; 
    int height ; 
    int depth ; 
    int channels ; 

    SIMG*      img ; 

    std::vector<float4> isect ; 
    std::vector<uchar4> pixel ; 
    std::vector<quad4>  photon ; 

    uchar4* d_pixel = nullptr ; 
    float4* d_isect = nullptr ; 
    quad4*  d_photon = nullptr ; 

    Frame(int width, int height, int depth );
 
    void init(); 
    void download(); 

    void annotate( const char* bottom_line=nullptr, const char* top_line=nullptr, int line_height=24  );

    void write(const char* outdir, int jpg_quality) const ;
    void writePNG(const char* dir, const char* name) const ;
    void writeJPG(const char* dir, const char* name, int quality) const ;

    void writePNG(const char* path) const ;
    void writeJPG(const char* path, int quality) const ;

    void writeIsect(  const char* dir, const char* name) const ;
    void writePhoton( const char* dir, const char* name) const ;


    unsigned char* getPixelData() const ;
    float*         getIntersectData() const ;
    quad4*         getPhotonData() const ;

    uchar4* getDevicePixel() const ; 
    float4* getDeviceIsect() const ; 
    quad4*  getDevicePhoton() const ; 

}; 


