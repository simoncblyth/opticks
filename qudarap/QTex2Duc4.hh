#pragma once
#include <cstddef>
#include <texture_types.h>
#include "QUDARAP_API_EXPORT.hh"


struct QUDARAP_API QTex2Duc4
{
    size_t       width ; 
    size_t       height ; 
    const void*  src ;
    uchar4*      dst ; 
    uchar4*      d_dst ; 

    cudaArray*   cuArray ; 
    cudaChannelFormatDesc channelDesc ;
    cudaTextureObject_t texObj ;


    QTex2Duc4( size_t width, size_t height, const void* src );
    virtual ~QTex2Duc4();  

    void init(); 
    void createArray(); 
    void uploadToArray(); 
    void createTextureObject(); 
    void rotate(float theta);
};





