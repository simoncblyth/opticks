#pragma once
/**
SCU_.h
=========

TODO : merge this with SCU.h 

**/

#include <cuda_runtime.h>
#include <string>
#include <sstream>


struct SCU_Exception : public std::runtime_error
{
     SCU_Exception( const char* msg ) : std::runtime_error( msg ) {}
};  
    
#define SCU_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw SCU_Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )



struct SCU_
{
    using ULL = unsigned long long ; 
    static void _cudaMalloc( void** p2p, size_t size, const char* label );

    template <typename T>
    static T*   device_alloc( unsigned num_items, const char* label ) ;

    template<typename T>
    static void copy_host_to_device( T* d, const T* h, unsigned num_items); 

};


inline void SCU_::_cudaMalloc( void** p2p, size_t size, const char* label )
{
    cudaError_t err = cudaMalloc(p2p, size ) ;   
    if( err != cudaSuccess ) 
    {    
        std::stringstream ss; 
        ss << "CUDA call (" << label << " ) failed with error: '"    
           << cudaGetErrorString( err )    
           << "' (" __FILE__ << ":" << __LINE__ << ")\n";  

        throw SCU_Exception( ss.str().c_str() );    
    }    
}


template<typename T>
inline T* SCU_::device_alloc( unsigned num_items, const char* label )
{
    size_t size = num_items*sizeof(T) ; 

    T* d ;   
    _cudaMalloc( reinterpret_cast<void**>( &d ), size, label );  

    return d ; 
}


template<typename T>
inline void SCU_::copy_host_to_device( T* d, const T* h, unsigned num_items)
{
    size_t size = num_items*sizeof(T) ;
    SCU_CHECK( cudaMemcpy(reinterpret_cast<void*>( d ), h , size, cudaMemcpyHostToDevice ));
}


