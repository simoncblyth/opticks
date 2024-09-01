#pragma once
#include <cstddef>
#include <cstdint>
#include <vector_types.h>
#include <ostream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CUDA_CHECK.h"


template <typename T>
struct SCU_Buf
{
   T*          ptr ; 
   size_t      num_item ;
   std::string name ; 

   CUdeviceptr pointer() const ; 
   void free() ; 

   std::string desc() const ; 
};


template <typename T>
inline CUdeviceptr SCU_Buf<T>::pointer() const
{
   return (CUdeviceptr)(uintptr_t) ptr ; 
}

template <typename T>
inline void SCU_Buf<T>::free()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( ptr) ) );
    ptr = nullptr ; 
    num_item = 0 ; 
}


template <typename T>
inline std::string SCU_Buf<T>::desc() const
{
    std::stringstream ss ; 
    ss << "SCU_Buf"
       << " (uintptr_t)ptr  0x" 
       << std::setw(9) << std::hex << (uintptr_t)ptr << std::dec
       << " sizeof(T) " << std::setw(5) << sizeof(T)
       << " num_item "  << std::setw(7) << num_item 
       << " name " << name 
       ;
    std::string str = ss.str(); 
    return str ; 
}





struct SCU
{
    template<typename T>
    static CUdeviceptr DevicePointerCast( const T* d_ptr ); 

    void _cudaMalloc( void** p2p, size_t size, const char* label ); 

    template<typename T>
    T* device_alloc( unsigned num_items, const char* label ); 



    template <typename T>
    static T* UploadArray(const T* array, size_t num_item ); 

    template <typename T>
    static SCU_Buf<T> UploadBuf(const T* array, size_t num_item, const char* name ); 



    template <typename T>
    static T* DownloadArray(const T* d_array, size_t num_item ); 

    template <typename T>
    static T* DownloadBuf(const SCU_Buf<T>& buf ); 

    template <typename T>
    static void DownloadVec(std::vector<T>& vec, const T* d_array, unsigned num_items); 



    template <typename T>
    static void FreeArray(T* d_array ); 

    template <typename T>
    static void FreeBuf(SCU_Buf<T>& buf ); 



    static void ConfigureLaunch2D( dim3& numBlocks, dim3& threadsPerBlock, int32_t width, int32_t height ); 
};


template<typename T>
CUdeviceptr SCU::DevicePointerCast( const T* d_ptr ) // static
{
    return (CUdeviceptr) (uintptr_t) d_ptr ; 
}


inline void SCU::_cudaMalloc( void** p2p, size_t size, const char* label )
{
    cudaError_t err = cudaMalloc(p2p, size ) ;   
    if( err != cudaSuccess ) 
    {    
        std::stringstream ss; 
        ss << "CUDA call (" << label << " ) failed with error: '"    
           << cudaGetErrorString( err )    
           << "' (" __FILE__ << ":" << __LINE__ << ")\n"
           ;  

        const char* msg = ss.str().c_str() ;
        throw CUDA_Exception(msg);    
    }    
}


template<typename T>
inline T* SCU::device_alloc( unsigned num_items, const char* label )
{
    size_t size = num_items*sizeof(T) ; 

    T* d ;   
    _cudaMalloc( reinterpret_cast<void**>( &d ), size, label );  

    return d ; 
}




/**
SCU::UploadArray
-----------------

Allocate on device and copy from host to device

**/
template <typename T>
inline T* SCU::UploadArray(const T* array, size_t num_item ) // static
{
    T* d_array = nullptr ; 
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_array ), num_item*sizeof(T) )); 
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_array ), array, sizeof(T)*num_item, cudaMemcpyHostToDevice )); 
    return d_array ; 
}

template <typename T>
inline SCU_Buf<T> SCU::UploadBuf(const T* array, size_t num_item, const char* name )
{
    T* d_array = UploadArray<T>(array, num_item ) ;   
    return { d_array, num_item, name } ; 
}


/**
SCU::DownloadArray  
-------------------

Allocate on host and copy from device to host 

**/

template <typename T>
inline T* SCU::DownloadArray(const T* d_array, size_t num_items ) // static
{
    T* array = new T[num_items] ;
    CUDA_CHECK( cudaMemcpy( array, d_array, sizeof(T)*num_items, cudaMemcpyDeviceToHost ));
    return array ;
}

template <typename T>
inline T* SCU::DownloadBuf(const SCU_Buf<T>& buf )
{
    return DownloadArray<T>( buf.ptr, buf.num_item );     
}


/**
SCU::DownloadVec
-----------------

After CU::DownloadVec

**/

template <typename T>
inline void SCU::DownloadVec(std::vector<T>& vec, const T* d_array, unsigned num_items)  // static
{
    unsigned num_bytes = num_items*sizeof(T) ; 
    vec.clear(); 
    vec.resize(num_items); 
    CUDA_CHECK( cudaMemcpy( vec.data(), d_array, num_bytes, cudaMemcpyDeviceToHost )); 
} 

template void SCU::DownloadVec<float>(   std::vector<float>&    vec,  const float* d_array,    unsigned num_items) ;
template void SCU::DownloadVec<unsigned>(std::vector<unsigned>& vec,  const unsigned* d_array, unsigned num_items) ;






template <typename T>
inline void SCU::FreeArray(T* d_array ) // static
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_array ) ) );
} 

template <typename T>
inline void SCU::FreeBuf(SCU_Buf<T>& buf ) // static
{ 
    buf.free() ;  
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

