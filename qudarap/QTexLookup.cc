#include <sstream>
#include <cuda_runtime.h>

#include "SLOG.hh"
#include "scuda.h"
#include "NP.hh"

#include "QUDA_CHECK.h"
#include "QTex.hh"
#include "QTexLookup.hh"

template<typename T>
const plog::Severity QTexLookup<T>::LEVEL = SLOG::EnvLevel("QTexLookup", "DEBUG") ;


template<typename T>
NP* QTexLookup<T>::Look( const QTex<T>* tex_ )  // static
{
    QTexLookup<T> look(tex_) ;
    return look.lookup();
}


template<typename T>
QTexLookup<T>::QTexLookup( const QTex<T>* tex_ )
    :
    tex(tex_)
{
}

/**
QTexLookup::lookup
--------------------

First tried using float4 and float template specialization for this but gave linker errors.
Instead kludged *is_float4* from the size of the template type.

This needs a revisit if wish to get this working with uchar "image" textures.

**/

template<typename T> NP* QTexLookup<T>::lookup()
{
    unsigned nx = tex->width ;
    unsigned ny = tex->height ;
    bool is_float4 = sizeof(T) == 4*sizeof(float);

    NP* out = NP::Make<float>(ny, nx, is_float4 ? 4 : 1 ) ;
    float* out_v = out->values<float>();

    lookup_populate( (T*)out_v ) ;

    return out ;
}



template <typename T>
extern void QTexLookup_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, T* lookup );


template<typename T>
void QTexLookup<T>::lookup_populate( T* lookup )
{
    LOG(LEVEL) << "[" ;
    size_t size = tex->width*tex->height*sizeof(T) ;

    LOG(LEVEL)
        << " tex.width " << tex->width
        << " tex.height " << tex->height
        << " size " << size
        << " tex->texObj " << tex->texObj
        << " tex->meta " << tex->meta
        << " tex->d_meta " << tex->d_meta
        ;

    dim3 numBlocks ;
    dim3 threadsPerBlock ;
    configureLaunch( numBlocks, threadsPerBlock, tex->width, tex->height );

    T* d_lookup = nullptr ;
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_lookup ), size ));

    QTexLookup_lookup<T>(numBlocks, threadsPerBlock, tex->texObj, tex->d_meta, (T*)d_lookup );

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( lookup ), d_lookup, size, cudaMemcpyDeviceToHost ));
    QUDA_CHECK( cudaFree(d_lookup) );

    cudaDeviceSynchronize();

    LOG(LEVEL) << "]" ;
}



template<typename T>
void QTexLookup<T>::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 16 ;
    threadsPerBlock.y = 16 ;
    threadsPerBlock.z = 1 ;

    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ;
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ;

    LOG(LEVEL)
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
        ;
}


template struct QUDARAP_API QTexLookup<float4> ;
template struct QUDARAP_API QTexLookup<float> ;



