#include "scuda.h"
#include "QUDA_CHECK.h"
#include "QU.hh"
#include "curand_kernel.h"
#include "qctx.h"
#include "qprop.h"


/**
QU::UploadArray
----------------

Allocate on device and copy from host to device

**/

template <typename T>
T* QU::UploadArray(const T* array, unsigned num_items ) // static
{
    T* d_array = nullptr ; 
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_array ), num_items*sizeof(T) )); 
    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_array ), array, sizeof(T)*num_items, cudaMemcpyHostToDevice )); 
    return d_array ; 
}

/**
QU::DownloadArray  
-------------------

Allocate on host and copy from device to host 

**/

template <typename T>
T* QU::DownloadArray(const T* d_array, unsigned num_items ) // static
{
    T* array = new T[num_items] ;   
    QUDA_CHECK( cudaMemcpy( array, d_array, sizeof(T)*num_items, cudaMemcpyDeviceToHost )); 
    return array ; 
}


template float* QU::UploadArray<float>(const float* array, unsigned num_items) ;
template float* QU::DownloadArray<float>(const float* d_array, unsigned num_items) ;

template unsigned* QU::UploadArray<unsigned>(const unsigned* array, unsigned num_items) ;
template unsigned* QU::DownloadArray<unsigned>(const unsigned* d_array, unsigned num_items) ;

template quad4* QU::UploadArray<quad4>(const quad4* array, unsigned num_items) ;
template quad4* QU::DownloadArray<quad4>(const quad4* d_array, unsigned num_items) ;

template curandState* QU::UploadArray<curandState>(const curandState* array, unsigned num_items) ;
template curandState* QU::DownloadArray<curandState>(const curandState* d_array, unsigned num_items) ;

template qctx* QU::UploadArray<qctx>(const qctx* array, unsigned num_items) ;
template qctx* QU::DownloadArray<qctx>(const qctx* d_array, unsigned num_items) ;

template qprop<float>* QU::UploadArray<qprop<float>>(const qprop<float>* array, unsigned num_items) ;
template qprop<float>* QU::DownloadArray<qprop<float>>(const qprop<float>* d_array, unsigned num_items) ;

template qprop<double>* QU::UploadArray<qprop<double>>(const qprop<double>* array, unsigned num_items) ;
template qprop<double>* QU::DownloadArray<qprop<double>>(const qprop<double>* d_array, unsigned num_items) ;








