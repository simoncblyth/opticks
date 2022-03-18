
#include <sstream>
#include <cstring>
#include <cassert>

#include "scuda.h"
#include "squad.h"

#include <cuda_runtime.h>
#include <iostream>
#include "cudaCheckErrors.h"
#include "QU.hh"
#include "QTex.hh"


template<typename T>
QTex<T>::QTex(size_t width_, size_t height_ , const void* src_, char filterMode_, bool normalizedCoords_ )
    :   
    width(width_),
    height(height_),
    src(src_),
    filterMode(filterMode_),
    normalizedCoords(normalizedCoords_), 
    origin(nullptr),

    cuArray(nullptr),
    channelDesc(cudaCreateChannelDesc<T>()),
    texObj(0),
    meta(new quad4),
    d_meta(nullptr)
{
    init(); 
}

template<typename T>
void QTex<T>::setOrigin(const void* origin_) 
{
    origin = origin_  ; 
}
template<typename T>
const void* QTex<T>::getOrigin() const  
{
    return origin ; 
}

template<typename T>
void QTex<T>::setHDFactor(unsigned hd_factor) 
{
    meta->q0.u.w = hd_factor ; 
}

template<typename T>
unsigned QTex<T>::getHDFactor() const 
{
    return meta->q0.u.w ; 
}

template<typename T>
char QTex<T>::getFilterMode() const 
{
    return filterMode ; 
}

template<typename T>
bool QTex<T>::getNormalizedCoords() const 
{
    return normalizedCoords ; 
}





template<typename T>
QTex<T>::~QTex()
{
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
}

template<typename T>
void QTex<T>::init()
{
    createArray();   // cudaMallocArray using channelDesc for T 
    uploadToArray();
    createTextureObject();

    meta->q0.u.x = width ; 
    meta->q0.u.y = height ; 
    meta->q0.u.z = 0 ; 
    meta->q0.u.w = 0 ; 
}


template<typename T>
void QTex<T>::setMetaDomainX( const quad* domx )
{
    meta->q1.f.x = domx->f.x ; 
    meta->q1.f.y = domx->f.y ; 
    meta->q1.f.z = domx->f.z ; 
    meta->q1.f.w = domx->f.w ; 
}

template<typename T>
void QTex<T>::setMetaDomainY( const quad* domy )
{
    meta->q2.f.x = domy->f.x ; 
    meta->q2.f.y = domy->f.y ; 
    meta->q2.f.z = domy->f.z ; 
    meta->q2.f.w = domy->f.w ; 
}


template<typename T>
std::string QTex<T>::desc() const
{
    std::stringstream ss ; 

    ss << "QTex"
       << " width " << width 
       << " height " << height 
       << " texObj " << texObj
       << " meta " << meta
       << " d_meta " << d_meta
       ;

    std::string s = ss.str(); 
    return s ; 
}

template<typename T>
void QTex<T>::createArray()
{
    cudaMallocArray(&cuArray, &channelDesc, width, height );
    cudaCheckErrors("cudaMallocArray");
}

/**
QTex::uploadToArray
----------------------

::

    cudaError_t 
    cudaMemcpy2DToArray(
       struct cudaArray* dst, 
       size_t wOffset, 
       size_t hOffset, 
       const void* src, 
       size_t spitch, 
       size_t width, 
       size_t height, 
       enum cudaMemcpyKind kind) 

Copies a matrix (height rows of width bytes each) from the memory area pointed to by src 
to the CUDA array dst starting at the upper left corner (wOffset, hOffset) where kind is one of 
cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice,
and specifies the direction of the copy. 
spitch is the width in memory in bytes of the 2D array pointed to by src, 
including any padding added to the end of each row. 
wOffset + width must not exceed the width of the CUDA array dst. 
width must not exceed spitch. 

cudaMemcpy2DToArray() returns an error if spitch exceeds the maximum allowed.

dst - Destination memory address 
wOffset - Destination starting X offset
hOffset - Destination starting Y offset
src - Source memory address
spitch - Pitch of source memory
width - Width of matrix transfer (columns in bytes) 
height - Height of matrix transfer (rows)
kind - Type of transfer


* https://forums.developer.nvidia.com/t/cudamemcpytoarray-is-deprecated/71385/10

**/

template<typename T>
void QTex<T>::uploadToArray()
{
    cudaArray_t dst = cuArray ;
    size_t wOffset = 0 ;
    size_t hOffset = 0 ;
    cudaMemcpyKind kind = cudaMemcpyHostToDevice ;

    size_t spitch = width*sizeof(T);  
    size_t width_bytes = width*sizeof(T); 
    size_t height_rows = height ; 

    cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width_bytes, height_rows, kind );

    cudaCheckErrors("cudaMemcpy2DToArray");
}





template<typename T>
void QTex<T>::uploadMeta()
{
    // not doing this automatically as will need to add some more meta 
    d_meta = QU::UploadArray<quad4>(meta, 1 );  
}

/**

normalized:false
   means texel coordinate addressing 

normalized:true
   eg reemission generation need normalized

**/

template<typename T>
void QTex<T>::createTextureObject()
{
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaTextureDesc.html
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;

    assert( filterMode == 'P' || filterMode == 'L' ); 
    switch(filterMode)
    {
        case 'L': texDesc.filterMode = cudaFilterModeLinear ; break ; 
        case 'P': texDesc.filterMode = cudaFilterModePoint  ; break ;  // ModePoint: switches off interpolation, necessary with with char texture  
    }

    texDesc.readMode = cudaReadModeElementType;  // return data of the type of the underlying buffer
    texDesc.normalizedCoords = normalizedCoords ;   // addressing into the texture with floats in range 0:1 when true

    // Create texture object
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
}








/**
https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/

A group of threads is called a CUDA block. 
Each CUDA block is executed by one streaming multiprocessor.
CUDA architecture limits the numbers of threads per block (1024 threads per block limit).

CUDA blocks are grouped into a grid. 
A kernel is executed as a grid of blocks of threads::

    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

The below *numBlocks* divides by the *threadsPerBlock* to give sufficient threads to cover the workspace, 
potentially with some spare threads at edge when workspace is not an exact multiple of threadsPerBlock size.

**/

// API export is essential on this template struct, otherwise get all symbols missing 
template struct QUDARAP_API QTex<uchar4>;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
// quell warning: type attributes ignored after type is already defined [-Wattributes]
template struct QUDARAP_API QTex<float>;
template struct QUDARAP_API QTex<float4>;
#pragma GCC diagnostic pop


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////


