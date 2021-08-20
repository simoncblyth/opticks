#include <iostream>
#include "sutil_vec_math.h"    

#include "cuda_runtime.h"
#include "CUDA_CHECK.h"

#ifdef WITH_PLOG
#include "PLOG.hh"
#endif

#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"
#include "Quad.h"
#include "qat4.h"

#include "CU.h"

#ifdef WITH_PLOG
const plog::Severity CU::LEVEL = PLOG::EnvLevel("CU","DEBUG"); 
#endif

/**
CU::UploadArray
----------------

Allocate on device and copy from host to device

**/
template <typename T>
T* CU::UploadArray(const T* array, unsigned num_items ) // static
{
#ifdef WITH_PLOG
    LOG(LEVEL) << " num_items " << num_items  ; 
#endif
    T* d_array = nullptr ; 
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_array ), num_items*sizeof(T) ));
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_array ), array, sizeof(T)*num_items, cudaMemcpyHostToDevice ));
    return d_array ; 
}

/**
CU::UploadArray  
----------------

Allocate on host and copy from device to host 

**/

template <typename T>
T* CU::DownloadArray(const T* d_array, unsigned num_items ) // static
{
#ifdef WITH_PLOG
    LOG(LEVEL) << " num_items " << num_items  ; 
#endif
    T* array = new T[num_items] ;  
    CUDA_CHECK( cudaMemcpy( array, d_array, sizeof(T)*num_items, cudaMemcpyDeviceToHost ));
    return array ; 
}


template float* CU::UploadArray<float>(const float* array, unsigned num_items) ;
template float* CU::DownloadArray<float>(const float* d_array, unsigned num_items) ;

template unsigned* CU::UploadArray<unsigned>(const unsigned* array, unsigned num_items) ;
template unsigned* CU::DownloadArray<unsigned>(const unsigned* d_array, unsigned num_items) ;

template float4* CU::UploadArray<float4>(const float4* array, unsigned num_items) ;
template float4* CU::DownloadArray<float4>(const float4* d_array, unsigned num_items) ;

template CSGNode* CU::UploadArray<CSGNode>(const CSGNode* d_array, unsigned num_items) ;
template CSGNode* CU::DownloadArray<CSGNode>(const CSGNode* d_array, unsigned num_items) ;

template quad4* CU::UploadArray<quad4>(const quad4* d_array, unsigned num_items) ;
template quad4* CU::DownloadArray<quad4>(const quad4* d_array, unsigned num_items) ;

template qat4* CU::UploadArray<qat4>(const qat4* d_array, unsigned num_items) ;
template qat4* CU::DownloadArray<qat4>(const qat4* d_array, unsigned num_items) ;

template CSGPrim* CU::UploadArray<CSGPrim>(const CSGPrim* d_array, unsigned num_items) ;
template CSGPrim* CU::DownloadArray<CSGPrim>(const CSGPrim* d_array, unsigned num_items) ;

template CSGSolid* CU::UploadArray<CSGSolid>(const CSGSolid* d_array, unsigned num_items) ;
template CSGSolid* CU::DownloadArray<CSGSolid>(const CSGSolid* d_array, unsigned num_items) ;





template <typename T>
T* CU::UploadVec(const std::vector<T>& vec)
{
    unsigned num_items = vec.size() ; 
    unsigned num_bytes = num_items*sizeof(T) ; 
#ifdef WITH_PLOG
    LOG(LEVEL) << " num_items " << num_items  ; 
#endif
    T* d_array = nullptr ; 
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_array ), num_bytes ));
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_array ), vec.data(), num_bytes, cudaMemcpyHostToDevice ));
    return d_array ; 
}

template CSGPrim* CU::UploadVec<CSGPrim>(const std::vector<CSGPrim>& vec ) ;
template float*     CU::UploadVec<float>(const std::vector<float>& vec ) ;
template unsigned*  CU::UploadVec<unsigned>(const std::vector<unsigned>& vec ) ;


template <typename T>
void CU::DownloadVec(std::vector<T>& vec, const T* d_array, unsigned num_items)  // static
{
#ifdef WITH_PLOG
    LOG(LEVEL) << " num_items " << num_items ; 
#endif
    unsigned num_bytes = num_items*sizeof(T) ; 
    vec.clear(); 
    vec.resize(num_items); 
    CUDA_CHECK( cudaMemcpy( vec.data(), d_array, num_bytes, cudaMemcpyDeviceToHost ));
} 

template void CU::DownloadVec<CSGPrim>(std::vector<CSGPrim>& vec,  const CSGPrim* d_array, unsigned num_items) ;
template void CU::DownloadVec<float>(std::vector<float>& vec,  const float* d_array, unsigned num_items) ;
template void CU::DownloadVec<unsigned>(std::vector<unsigned>& vec,  const unsigned* d_array, unsigned num_items) ;



