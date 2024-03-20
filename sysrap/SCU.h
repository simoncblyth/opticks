#pragma once
#include <cstdint>
#include <vector_types.h>
#include <ostream>
#include <iomanip>
#include <cuda_runtime.h>

#include "CUDA_CHECK.h"

struct SCU
{
    template <typename T>
    static T* UploadArray(const T* array, unsigned num_items ); 

    template <typename T>
    static T* DownloadArray(const T* d_array, unsigned num_items ); 


    static void ConfigureLaunch2D( dim3& numBlocks, dim3& threadsPerBlock, int32_t width, int32_t height ); 
};



/**
SCU::UploadArray
-----------------

Allocate on device and copy from host to device

**/
template <typename T>
inline T* SCU::UploadArray(const T* array, unsigned num_items ) // static
{
    T* d_array = nullptr ; 
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_array ), num_items*sizeof(T) )); 
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_array ), array, sizeof(T)*num_items, cudaMemcpyHostToDevice )); 
    return d_array ; 
}


/**
SCU::DownloadArray  
-------------------

Allocate on host and copy from device to host 

**/

template <typename T>
inline T* SCU::DownloadArray(const T* d_array, unsigned num_items ) // static
{
    T* array = new T[num_items] ;
    CUDA_CHECK( cudaMemcpy( array, d_array, sizeof(T)*num_items, cudaMemcpyDeviceToHost ));
    return array ;
}



inline void SCU::ConfigureLaunch2D( dim3& numBlocks, dim3& threadsPerBlock, int32_t width, int32_t height ) // static
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 16 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}

inline std::ostream& operator<<(std::ostream& os, const dim3& v)
{
    int w = 6 ;
    os
       << "("
       << std::setw(w) << v.x
       << ","
       << std::setw(w) << v.y
       << ","
       << std::setw(w) << v.z
       << ") "
       ;
    return os;
}

