#pragma once
#include <cstddef>
#include <texture_types.h>
#include "QUDARAP_API_EXPORT.hh"


template<typename T>
struct QUDARAP_API QTex2D
{
    size_t       width ; 
    size_t       height ; 
    const void*  src ;
    T*           dst ; 
    T*           d_dst ; 

    cudaArray*   cuArray ; 
    cudaChannelFormatDesc channelDesc ;
    cudaTextureObject_t texObj ;


    QTex2D( size_t width, size_t height, const void* src );
    virtual ~QTex2D();  

    void init(); 
    void createArray(); 
    void uploadToArray(); 
    void createTextureObject(); 
    void rotate(float theta);
};





