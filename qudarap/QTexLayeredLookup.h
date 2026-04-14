#pragma once

/**
QTexLayeredLookup
==================

This provides a roundtrip test for a layered GPU texture, looking
up every texel in every layer of the texture.
The resulting lookup array should exactly match the input array.

**/


struct NP ;
struct dim3 ;
template<typename T> struct QTexLayered ;


template<typename T>
struct QTexLayeredLookup
{
    QTexLayeredLookup( const QTexLayered<T>* tex_ );
    const QTexLayered<T>* tex ;

    NP* lookup();

    void lookup_populate( T* lookup );
    void configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height );
};



#include <sstream>
#include <cuda_runtime.h>

#include "scuda.h"
#include "NP.hh"

#include "QTexLayered.h"
#include "QUDA_CHECK.h"


template<typename T>
QTexLayeredLookup<T>::QTexLayeredLookup( const QTexLayered<T>* tex_ )
    :
    tex(tex_)
{
}

/**
QTexLayeredLookup::lookup
---------------------------

First tried using float4 and float template specialization for this but gave linker errors.
Instead kludged *is_float4* from the size of the template type.

This needs a revisit if wish to get this working with uchar "image" textures.

**/

template<typename T> NP* QTexLayeredLookup<T>::lookup()
{
    unsigned nl = tex->layers ;
    unsigned ny = tex->height ;
    unsigned nx = tex->width ;
    unsigned payload = tex->payload ;

    NP* out = NP::Make<float>(nl, ny, nx, payload ) ;
    float* out_v = out->values<float>();

    lookup_populate( (T*)out_v ) ;

    return out ;
}



template <typename T>
extern void QTexLayeredLookup_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, T* lookup, unsigned layer );


template<typename T>
void QTexLayeredLookup<T>::lookup_populate( T* lookup )
{
    size_t size = tex->layers*tex->width*tex->height*sizeof(T) ;

    std::cout
        << " tex.layers " << tex->layers
        << " tex.width " << tex->width
        << " tex.height " << tex->height
        << " tex.payload " << tex->payload
        << " size " << size
        << " tex->tex " << tex->tex
        << " tex->meta " << tex->meta
        << " tex->d_meta " << tex->d_meta
        << "\n"
        ;

    dim3 numBlocks ;
    dim3 threadsPerBlock ;
    configureLaunch( numBlocks, threadsPerBlock, tex->width, tex->height );

    T* d_lookup = nullptr ;
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_lookup ), size ));


    for(unsigned layer ; layer < tex->layers ; layer++ )
    {
        std::cout << "QTexLayeredLookup::lookup_populate layer " << layer << "\n" ;
        QTexLayeredLookup_lookup<T>(numBlocks, threadsPerBlock, tex->tex, tex->d_meta, (T*)d_lookup, layer );
    }

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( lookup ), d_lookup, size, cudaMemcpyDeviceToHost ));
    QUDA_CHECK( cudaFree(d_lookup) );

    cudaDeviceSynchronize();

}



template<typename T>
void QTexLayeredLookup<T>::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 16 ;
    threadsPerBlock.y = 16 ;
    threadsPerBlock.z = 1 ;

    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ;
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ;

    std::cout
        << " width " << std::setw(7) << width
        << " height " << std::setw(7) << height
        << " width*height " << std::setw(7) << width*height
        << " threadsPerBlock"
        << "("
        << std::setw(3) << threadsPerBlock.x << " "
        << std::setw(3) << threadsPerBlock.y << " "
        << std::setw(3) << threadsPerBlock.z << " "
        << ")"
        << " numBlocks "
        << "("
        << std::setw(3) << numBlocks.x << " "
        << std::setw(3) << numBlocks.y << " "
        << std::setw(3) << numBlocks.z << " "
        << ")"
        << "\n"
        ;
}


//template struct QTexLayeredLookup<float4> ;
template struct QTexLayeredLookup<float> ;



