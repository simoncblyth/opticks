
#pragma once

#include <optix.h>
#include <stdexcept>
#include <sstream>
#include <iostream>


//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------


#define OPTIX_CHECK( call )                                                     \
    do                                                                          \
    {                                                                           \
        OptixResult res = call;                                                 \
        if( res != OPTIX_SUCCESS )                                              \
        {                                                                       \
            std::stringstream ss;                                               \
            ss << "OPTIX_CHECK "                                                \
               << " call : '" << #call << "' failed: "                          \
               << "(" << __FILE__ ":" << __LINE__ << ")\n"                      \
               << optixGetErrorName(res) << "\n"                                \
               << "\n" ;                                                        \
            throw OPTIX_Exception( res, ss.str().c_str() );                     \
        }                                                                       \
    } while( 0 )


// NOTHOW variant can be used from dtor to avoid warnings
#define OPTIX_CHECK_NOTHROW( call )                                             \
    do                                                                          \
    {                                                                           \
        OptixResult res = call;                                                 \
        if( res != OPTIX_SUCCESS )                                              \
        {                                                                       \
            std::stringstream ss;                                               \
            ss << "OPTIX_CHECK_NOTHROW call '" << #call << "' failed: "         \
               << "(" << __FILE__ ":" << __LINE__ << ")\n"                      \
               << optixGetErrorName(res) << "\n"                                \
               << "\n" ;                                                        \
            std::cerr << ss.str();                                              \
        }                                                                       \
    } while( 0 )





#define OPTIX_CHECK_LOG( call )                                                \
    do                                                                         \
    {                                                                          \
        OptixResult res = call;                                                \
        if( res != OPTIX_SUCCESS )                                             \
        {                                                                      \
            std::stringstream ss;                                              \
            ss << "Optix call '" << #call << "' failed: " __FILE__ ":"         \
               << __LINE__ << ")\nLog:\n" << log                               \
               << ( sizeof_log > sizeof( log ) ? "<TRUNCATED>" : "" )          \
               << "\n";                                                        \
            throw OPTIX_Exception( res, ss.str().c_str() );                   \
        }                                                                      \
    } while( 0 )




class OPTIX_Exception : public std::runtime_error
{
 public:
     OPTIX_Exception( OptixResult res, const char* msg )
         : std::runtime_error( createMessage( res, msg ).c_str() )
     { }

 private:
     std::string createMessage( OptixResult res, const char* msg )
     {
         std::ostringstream out;
         out << optixGetErrorName( res ) << ": " << msg;
         return out.str();
     }
};



