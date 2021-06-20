#include <cuda_runtime.h>
#include <iostream>
#include "cudaCheckErrors.h"
#include "QTex.hh"

template<typename T>
QTex<T>::QTex(size_t width_, size_t height_ , const void* src_)
    :   
    width(width_),
    height(height_),
    src(src_),
    dst(new T[width*height]),
    d_dst(nullptr),
    cuArray(nullptr),
    channelDesc(cudaCreateChannelDesc<T>()),
    texObj(0)
{
    init(); 
}

template<typename T>
QTex<T>::~QTex()
{
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    delete[] dst ; 
    cudaFree(d_dst);
}

template<typename T>
void QTex<T>::init()
{
    createArray();
    uploadToArray();
    createTextureObject();
}

template<typename T>
void QTex<T>::createArray()
{
    cudaMallocArray(&cuArray, &channelDesc, width, height );
    cudaCheckErrors("cudaMallocArray");
}

template<typename T>
void QTex<T>::uploadToArray()
{
    cudaArray_t dst = cuArray ;
    size_t wOffset = 0 ;
    size_t hOffset = 0 ;
    size_t count = width*height*sizeof(T) ;
    cudaMemcpyKind kind = cudaMemcpyHostToDevice ;
    cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind );
    cudaCheckErrors("cudaMemcpyToArray");
}

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

    //texDesc.filterMode = cudaFilterModeLinear;
    texDesc.filterMode = cudaFilterModePoint;    // switch off interpolation, as that gives error with non-float texture  

    texDesc.readMode = cudaReadModeElementType;  // return data of the type of the underlying buffer
    texDesc.normalizedCoords = 1 ;            // addressing into the texture with floats in range 0:1

    // Create texture object
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
}


extern "C" void QTex_uchar4_rotate_kernel(dim3 dimGrid, dim3 dimBlock, uchar4* d_output, cudaTextureObject_t texObj,  size_t width, size_t height, float theta );

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

template<typename T>
void QTex<T>::rotate(float theta)
{
    cudaMalloc(&d_dst, width*height*sizeof(T));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    QTex_uchar4_rotate_kernel( numBlocks, threadsPerBlock, d_dst, texObj, width, height, theta );

    cudaDeviceSynchronize();
    cudaCheckErrors("cudaDeviceSynchronize");
    // Fatal error: cudaDeviceSynchronize (linear filtering not supported for non-float type at SIMGStandaloneTest.cu:123)

    cudaMemcpy(dst, d_dst, width*height*sizeof(T), cudaMemcpyDeviceToHost);
}


/**
Do nothing template specialization for float textures, rotation is only relevant to uchar4 2d images
**/
template<>  void QTex<float>::rotate(float theta)
{
}


// API export is essential on this template struct, otherwise get all symbols missing 
template struct QUDARAP_API QTex<uchar4>;
template struct QUDARAP_API QTex<float>;


