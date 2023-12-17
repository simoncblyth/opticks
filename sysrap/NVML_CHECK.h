#pragma once

#include <stdexcept>
#include <sstream>

#define NVML_CHECK( call )                                                     \
    do                                                                         \
    {                                                                          \
        nvmlReturn_t result = call;                                            \
        if( result != NVML_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "NVML call (" << #call << " ) failed with error: '"          \
               << nvmlErrorString( result )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw NVML_Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )



#define NVML_CHECK_RC( RC, call )                                                     \
    do                                                                         \
    {                                                                          \
        nvmlReturn_t result = call;                                            \
        if( result != RC )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "NVML call (" << #call << " ) failed with error: '"          \
               << nvmlErrorString( result )                                  \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n";                  \
            throw NVML_Exception( ss.str().c_str() );                        \
        }                                                                      \
    } while( 0 )








struct NVML_Exception : public std::runtime_error
{
    NVML_Exception( const char* msg ) : std::runtime_error( msg ){}
};

