#pragma once


#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#define _xstr(s) _str(s)
#define _str(s) #s

#include "CUDA_CHECK.h"
#include "OPTIX_CHECK.h"

#include "SOPTIX_Properties.h"

struct SOPTIX
{  
    static std::string DescAccelBufferSizes( const OptixAccelBufferSizes& accelBufferSizes ); 
    static std::string DescBuildInputTriangleArray( const OptixBuildInput& buildInput ); 
    static void LogCB( unsigned int level, const char* tag, const char* message, void* /*cbdata */) ;     

    const char* VERSION ; 
    OptixDeviceContext context ; 
    SOPTIX_Properties* props ; 

    std::string desc() const ;

    SOPTIX(); 
    void init(); 
};

inline void SOPTIX::LogCB( unsigned int level, const char* tag, const char* message, void* /*cbdata */)  // static 
{   
    std::stringstream ss ; 
    ss << "[" << std::setw(2) << level << "][" << std::setw( 12 ) << tag << "]: " << message << std::endl ;
    std::string line = ss.str() ;
    std::cout << line ;
}

inline std::string SOPTIX::desc() const
{
    std::stringstream ss ; 
    ss << "[ SOPTIX::desc \n" ; 
    ss << " OPTIX_VERSION " << OPTIX_VERSION << "\n" ;
    ss << props->desc() ; 
    ss << "] SOPTIX::desc \n" ; 
    std::string str = ss.str(); 
    return str ; 
}

inline SOPTIX::SOPTIX()
    :
    context(nullptr),
    props(nullptr),
    VERSION(_xstr(OPTIX_VERSION))
{
    init();
}

/**
SOPTIX::init
------------

Initialize CUDA and create OptiX context

**/

inline void SOPTIX::init()
{
    CUDA_CHECK(cudaFree( 0 ) );

    CUcontext cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &SOPTIX::LogCB ;
    options.logCallbackLevel          = 4;
    //options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL ; 

    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    props = new SOPTIX_Properties(context); 
}


inline std::string SOPTIX::DescAccelBufferSizes( const OptixAccelBufferSizes& accelBufferSizes ) // static
{
    std::stringstream ss ;  
    ss
        << "[SOPTIX::DescAccelBufferSizes"
        << std::endl
        << "accelBufferSizes.outputSizeInBytes     : " << accelBufferSizes.outputSizeInBytes
        << std::endl 
        << "accelBufferSizes.tempSizeInBytes       : " << accelBufferSizes.tempSizeInBytes
        << std::endl 
        << "accelBufferSizes.tempUpdateSizeInBytes : " << accelBufferSizes.tempUpdateSizeInBytes
        << std::endl 
        << "]SOPTIX::DescAccelBufferSizes"
        << std::endl 
        ; 
    std::string str = ss.str(); 
    return str ;
}

inline std::string SOPTIX::DescBuildInputTriangleArray( const OptixBuildInput& buildInput ) 
{
    std::stringstream ss ; 
    ss << "[SOPTIX::DescBuildInputTriangleArray" << std::endl ; 
    ss << " buildInput.triangleArray.numVertices      : " << buildInput.triangleArray.numVertices << std::endl ; 
    ss << " buildInput.triangleArray.numIndexTriplets : " << buildInput.triangleArray.numIndexTriplets << std::endl ; 
    ss << " buildInput.triangleArray.flags[0]         : " << buildInput.triangleArray.flags[0] << std::endl ; 
    ss << "]SOPTIX::DescBuildInputTriangleArray" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}




