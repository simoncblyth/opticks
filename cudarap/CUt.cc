#include "CUDA_CHECK.h"
#include "CUt.hh"
#include "curand_kernel.h"


/**
CUt::UploadArray
----------------

Allocate on device and copy from host to device

**/

template <typename T>
T* CUt::UploadArray(const T* array, unsigned num_items ) // static
{
    T* d_array = nullptr ; 
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_array ), num_items*sizeof(T) )); 
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_array ), array, sizeof(T)*num_items, cudaMemcpyHostToDevice )); 
    return d_array ; 
}

/**
CUt::DownloadArray  
-------------------

Allocate on host and copy from device to host 

**/

template <typename T>
T* CUt::DownloadArray(const T* d_array, unsigned num_items ) // static
{
    T* array = new T[num_items] ;   
    CUDA_CHECK( cudaMemcpy( array, d_array, sizeof(T)*num_items, cudaMemcpyDeviceToHost )); 
    return array ; 
}


template float* CUt::UploadArray<float>(const float* array, unsigned num_items) ;
template float* CUt::DownloadArray<float>(const float* d_array, unsigned num_items) ;

template unsigned* CUt::UploadArray<unsigned>(const unsigned* array, unsigned num_items) ;
template unsigned* CUt::DownloadArray<unsigned>(const unsigned* d_array, unsigned num_items) ;

template curandState* CUt::UploadArray<curandState>(const curandState* array, unsigned num_items) ;
template curandState* CUt::DownloadArray<curandState>(const curandState* d_array, unsigned num_items) ;



