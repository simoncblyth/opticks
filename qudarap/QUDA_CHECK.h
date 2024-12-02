#pragma once


#include "QUDARAP_API_EXPORT.hh"
#include <stdexcept>
#include <sstream>


#define QUDA_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        cudaError_t error = call;                                              \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA call (" << #call << " ) failed with error: '"          \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw QUDA_Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )


#define QUDA_SYNC_CHECK()                                                      \
    do                                                                         \
    {                                                                          \
        cudaDeviceSynchronize();                                               \
        cudaError_t error = cudaGetLastError();                                \
        if( error != cudaSuccess )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "CUDA error on synchronize with error '"                     \
               << cudaGetErrorString( error )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw QUDA_Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )




class QUDARAP_API QUDA_Exception : public std::runtime_error
{
 public:
     QUDA_Exception( const char* msg )
         : std::runtime_error( msg )
     { }

};


struct QUDA
{
    static void  before_kernel( cudaEvent_t& start, cudaEvent_t& stop ); 
    static float after_kernel(  cudaEvent_t& start, cudaEvent_t& stop );
}; 

inline void QUDA::before_kernel( cudaEvent_t& start, cudaEvent_t& stop )
{
    QUDA_CHECK( cudaEventCreate( &start ) );
    QUDA_CHECK( cudaEventCreate( &stop ) );
    QUDA_CHECK( cudaEventRecord( start,0 ) );
}

inline float QUDA::after_kernel( cudaEvent_t& start, cudaEvent_t& stop )
{
    float kernel_time = 0.f ;

    QUDA_CHECK( cudaEventRecord( stop,0 ) );
    QUDA_CHECK( cudaEventSynchronize(stop) );

    QUDA_CHECK( cudaEventElapsedTime(&kernel_time, start, stop) );
    QUDA_CHECK( cudaEventDestroy( start ) );
    QUDA_CHECK( cudaEventDestroy( stop ) );

    QUDA_CHECK( cudaDeviceSynchronize() );

    return kernel_time ;
}




