#include <cuda_runtime.h>
#include <iostream>
#include "cudaCheckErrors.h"
#include "QTex2D.hh"

template<typename T>
QTex2D<T>::QTex2D(size_t width_, size_t height_ , const void* src_)
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
QTex2D<T>::~QTex2D()
{
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    delete[] dst ; 
    cudaFree(d_dst);
}

template<typename T>
void QTex2D<T>::init()
{
    createArray();
    uploadToArray();
    createTextureObject();
}

template<typename T>
void QTex2D<T>::createArray()
{
    cudaMallocArray(&cuArray, &channelDesc, width, height );
    cudaCheckErrors("cudaMallocArray");
}

template<typename T>
void QTex2D<T>::uploadToArray()
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
void QTex2D<T>::createTextureObject()
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


extern "C" void transformKernel(dim3 dimGrid, dim3 dimBlock, uchar4* d_output, cudaTextureObject_t texObj,  size_t width, size_t height, float theta ); 
//extern "C" void transformKernel(dim3 dimGrid, dim3 dimBlock,  float* d_output, cudaTextureObject_t texObj,  size_t width, size_t height, float theta ); 


template<typename T>
void QTex2D<T>::rotate(float theta)
{
    cudaMalloc(&d_dst, width*height*sizeof(T));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    transformKernel( dimGrid, dimBlock, d_dst, texObj, width, height, theta );

    cudaDeviceSynchronize();
    cudaCheckErrors("cudaDeviceSynchronize");
    // Fatal error: cudaDeviceSynchronize (linear filtering not supported for non-float type at SIMGStandaloneTest.cu:123)

    cudaMemcpy(dst, d_dst, width*height*sizeof(T), cudaMemcpyDeviceToHost);
}


//template struct QTex2D<float>;
template struct QTex2D<uchar4>;


