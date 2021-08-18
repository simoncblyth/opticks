#pragma once


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
            throw sutil::QUDA_Exception( ss.str().c_str() );                        \
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
            throw sutil::QUDA_Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )



namespace sutil
{

class QUDA_Exception : public std::runtime_error
{
 public:
     QUDA_Exception( const char* msg )
         : std::runtime_error( msg )
     { }

};



} // end namespace sutil
