#pragma once

#include <string>
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
    char         filterMode ;  // 'L':cudaFilterModeLinear OR 'P':cudaFilterModePoint 

    T*           dst ; 
    T*           d_dst ; 

    cudaArray*   cuArray ; 
    cudaChannelFormatDesc channelDesc ;
    cudaTextureObject_t texObj ;

    quad4*              meta ; 
    quad4*              d_meta ; 

    QTex( size_t width, size_t height, const void* src, char filterMode  );

    void     setHDFactor(unsigned hd_factor_) ; 
    unsigned getHDFactor() const ; 
    char     getFilterMode() const ; 

    virtual ~QTex();  

    void init(); 
    std::string desc() const ; 

    void createArray(); 
    void uploadToArray(); 
    void uploadMeta(); 

    void createTextureObject(); 
    void rotate(float theta);
};





