#pragma once
/**
SOPTIX_Context.h : OptixDeviceContext + SOPTIX_Properties
==========================================================

**/

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#define _xstr(s) _str(s)
#define _str(s) #s

#include "CUDA_CHECK.h"
#include "OPTIX_CHECK.h"

#include "SOPTIX_Properties.h"

struct SOPTIX_Context
{  
    static void LogCB( unsigned int level, const char* tag, const char* message, void* /*cbdata */) ;     

    const char* VERSION ; 
    OptixDeviceContext context ; 
    SOPTIX_Properties* props ; 

    std::string desc() const ;

    SOPTIX_Context(); 
    void init(); 
};

inline void SOPTIX_Context::LogCB( unsigned int level, const char* tag, const char* message, void* /*cbdata */)  // static 
{   
    std::stringstream ss ; 
    ss << "[" << std::setw(2) << level << "][" << std::setw( 12 ) << tag << "]: " << message << std::endl ;
    std::string line = ss.str() ;
    std::cout << line ;
}

inline std::string SOPTIX_Context::desc() const
{
    std::stringstream ss ; 
    ss << "[ SOPTIX_Context::desc \n" ; 
    ss << " OPTIX_VERSION " << OPTIX_VERSION << "\n" ;
    ss << props->desc() ; 
    ss << "] SOPTIX_Context::desc \n" ; 
    std::string str = ss.str(); 
    return str ; 
}

inline SOPTIX_Context::SOPTIX_Context()
    :
    VERSION(_xstr(OPTIX_VERSION)),
    context(nullptr),
    props(nullptr)
{
    init();
}

/**
SOPTIX_Context::init
---------------------

Initialize CUDA and create OptiX context

**/

inline void SOPTIX_Context::init()
{
    CUDA_CHECK(cudaFree( 0 ) );

    OPTIX_CHECK( optixInit() );

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &SOPTIX_Context::LogCB ;
    options.logCallbackLevel          = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL ; 

    CUcontext cuCtx = 0;  // zero means take the current context

    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    props = new SOPTIX_Properties(context); 
}




