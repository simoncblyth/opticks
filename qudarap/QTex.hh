#pragma once
/**
QTex.hh
========


**/
#include <string>
#include <cstddef>
#include "scuda.h"

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
#include <texture_types.h>
#endif


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
    bool         normalizedCoords ; 
    const void*  origin ;  // typically an NP array 

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
    cudaArray*   cuArray ; 
    cudaChannelFormatDesc channelDesc ;
#endif
    cudaTextureObject_t texObj ;

    quad4*              meta ; 
    quad4*              d_meta ; 

    QTex( size_t width, size_t height, const void* src, char filterMode, bool normalizedCoords  );

    void     setMetaDomainX( const quad* domx ); 
    void     setMetaDomainY( const quad* domy ); 
    void     uploadMeta(); 

    void           setOrigin(const void* origin_) ; 
    const void*    getOrigin() const ; 

    void     setHDFactor(unsigned hd_factor_) ; 
    unsigned getHDFactor() const ; 

    char     getFilterMode() const ; 
    bool     getNormalizedCoords() const ; 

    virtual ~QTex();  

    void init(); 
    std::string desc() const ; 

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
    void createArray(); 
    void uploadToArray(); 
    void createTextureObject(); 
#endif

};


