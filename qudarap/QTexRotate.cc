#include <cuda_runtime.h>
#include "cudaCheckErrors.h"

#include "QTex.hh"
#include "QTexRotate.hh"

template<typename T>
QTexRotate<T>::QTexRotate(const QTex<T>* tex_) 
    :
    tex(tex_),
    rotate_dst(nullptr),
    d_rotate_dst(nullptr)
{
}

template<typename T>
QTexRotate<T>::~QTexRotate()
{
    delete[] rotate_dst ; 
    cudaFree(d_rotate_dst);
}

extern "C" void QTex_uchar4_rotate_kernel(dim3 dimGrid, dim3 dimBlock, uchar4* d_output, cudaTextureObject_t texObj,  size_t width, size_t height, float theta );

template<typename T>
void QTexRotate<T>::rotate(float theta)
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 


    cudaMalloc(&d_rotate_dst, width*height*sizeof(T));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    QTex_uchar4_rotate_kernel( numBlocks, threadsPerBlock, d_rotate_dst, tex->texObj, width, height, theta );

    cudaDeviceSynchronize();
    cudaCheckErrors("cudaDeviceSynchronize");
    // Fatal error: cudaDeviceSynchronize (linear filtering not supported for non-float type at SIMGStandaloneTest.cu:123)

    if(rotate_dst == nullptr)
    {
        rotate_dst = new T[width*height] ;  
    }

    cudaMemcpy(rotate_dst, d_rotate_dst, width*height*sizeof(T), cudaMemcpyDeviceToHost);
}



template struct QUDARAP_API QTexRotate<uchar4>;


