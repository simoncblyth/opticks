#pragma once
#include <cstddef>
#include <texture_types.h>
struct quad4 ; 

#include "QUDARAP_API_EXPORT.hh"


template<typename T>
struct QUDARAP_API QTex
{
    size_t       width ; 
    size_t       height ; 
    const void*  src ;
    T*           dst ; 
    T*           d_dst ; 

    cudaArray*   cuArray ; 
    cudaChannelFormatDesc channelDesc ;
    cudaTextureObject_t texObj ;

    quad4*              meta ; 
    quad4*              d_meta ; 


    QTex( size_t width, size_t height, const void* src );
    virtual ~QTex();  

    void init(); 
    void createArray(); 
    void uploadToArray(); 
    void uploadMeta(); 

    void createTextureObject(); 
    void rotate(float theta);
};





