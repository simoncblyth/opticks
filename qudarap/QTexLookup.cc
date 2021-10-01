#include <sstream>
#include <cuda_runtime.h>

#include "PLOG.hh"
#include "SSys.hh"
#include "scuda.h"
#include "NP.hh"

#include "QUDA_CHECK.h"
#include "QTex.hh"
#include "QTexLookup.hh"

template<typename T>
const plog::Severity QTexLookup<T>::LEVEL = PLOG::EnvLevel("QTexLookup", "DEBUG") ; 
 

template<typename T>
QTexLookup<T>::QTexLookup( const QTex<T>* tex_ )
    :
    tex(tex_)
{
}

template<typename T> NP* QTexLookup<T>::lookup()
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 
    unsigned num_lookup = width*height ; 

    bool is_float4 = sizeof(T) == 4*sizeof(float); 

    NP* out = NP::Make<float>(height, width, is_float4 ? 4 : 1 ) ; 
    float* out_v = out->values<float>(); 

    lookup_( (T*)out_v , num_lookup, width, height ); 

    return out ; 
}

// tried using float4 and float template specialization for this but getting linker errors 


template <typename T>
extern void QTexLookup_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, T* lookup, unsigned num_lookup, unsigned width, unsigned height  ); 


template<typename T>
void QTexLookup<T>::lookup_( T* lookup, unsigned num_lookup, unsigned width, unsigned height  )
{
    LOG(LEVEL) << "[" ; 
    size_t size = width*height*sizeof(T) ; 

    LOG(LEVEL) 
        << " num_lookup " << num_lookup
        << " width " << width 
        << " height " << height
        << " size " << size 
        << " tex->texObj " << tex->texObj
        << " tex->meta " << tex->meta
        << " tex->d_meta " << tex->d_meta
        ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, width, height ); 
  
    T* d_lookup = nullptr ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_lookup ), size )); 

    QTexLookup_lookup<T>(numBlocks, threadsPerBlock, tex->texObj, tex->d_meta, (T*)d_lookup, num_lookup, width, height );  

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


 
