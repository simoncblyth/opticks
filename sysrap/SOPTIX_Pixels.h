#pragma once

#include "sppm.h"

struct SOPTIX_Pixels
{
    const SGLM& gm ;
    size_t num_pixel ; 
    size_t num_bytes ; 
    uchar4* d_pixels ;
    uchar4* pixels ; 

    SOPTIX_Pixels(const SGLM& gm ); 
    void init(); 

    void download(); 
    void save_ppm(const char* path); 
};

inline SOPTIX_Pixels::SOPTIX_Pixels(const SGLM& _gm )
    :
    gm(_gm),
    num_pixel(gm.Width_Height()),
    num_bytes(num_pixel*sizeof(uchar4)),
    d_pixels(nullptr),
    pixels(new uchar4[num_pixel]) 
{
    init(); 
} 

inline void SOPTIX_Pixels::init()
{
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_pixels ), num_bytes )); 
} 
inline void SOPTIX_Pixels::download() 
{
    CUDA_CHECK( cudaMemcpy( pixels, reinterpret_cast<void*>(d_pixels), num_bytes, cudaMemcpyDeviceToHost ));
}    
inline void SOPTIX_Pixels::save_ppm(const char* path)
{
    bool yflip = true ; 
    int ncomp = 4 ; 
    sppm::Write(path, gm.Width(), gm.Height(), ncomp, (unsigned char*)pixels, yflip );  
}

