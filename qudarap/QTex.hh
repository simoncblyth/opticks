#pragma once
/**
QTex
=====


**/
#include <string>
#include <cstddef>
#include <texture_types.h>
struct quad4 ; 
union quad ; 

#include "QUDARAP_API_EXPORT.hh"


template<typename T>
struct QUDARAP_API QTex
{
    size_t       width ; 
    size_t       height ; 
    const void*  src ;
    char         filterMode ;  // 'L':cudaFilterModeLinear OR 'P':cudaFilterModePoint 
    const void*  origin ;  // typically an NP array 


    // TODO: split off rotation elsewhere "QTexRotate?", it is not typically needed 
    T*           rotate_dst ; 
    T*           d_rotate_dst ; 


    cudaArray*   cuArray ; 
    cudaChannelFormatDesc channelDesc ;
    cudaTextureObject_t texObj ;

    quad4*              meta ; 
    quad4*              d_meta ; 

    QTex( size_t width, size_t height, const void* src, char filterMode  );

    void     setMetaDomainX( const quad* domx ); 
    void     setMetaDomainY( const quad* domy ); 

    void           setOrigin(const void* origin_) ; 
    const void*    getOrigin() const ; 

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


