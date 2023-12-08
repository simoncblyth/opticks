
#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "Ctx.h"

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

#include "CUDA_CHECK.h"
#include "OPTIX_CHECK.h"
#include "Properties.h"

#include "ssys.h"
#include "SLOG.hh"

const plog::Severity Ctx::LEVEL = SLOG::EnvLevel("Ctx", "DEBUG") ; 


OptixDeviceContext Ctx::context = nullptr ;

void Ctx::log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)  // static 
{
    std::stringstream ss ; 
    ss << "[" << std::setw(2) << level << "][" << std::setw( 12 ) << tag << "]: " << message ;
    std::string line = ss.str() ; 
    log.push_back(line); 
    LOG(LEVEL) << line ; 
}

Ctx::Ctx()
    :
    props(nullptr)
{
    //CUDA_CHECK(cudaSetDevice(device)); // TOO LATE TO DO THIS HERE AS GEOM ALREADY UPLOADED
    CUDA_CHECK(cudaFree( 0 ) ); 

    CUcontext cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &Ctx::log_cb;
    options.logCallbackLevel          = 4;
    //options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL ; 

    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    props = new Properties ;   // instanciation gets the properties
}


std::string Ctx::desc() const 
{
    std::stringstream ss ; 
    ss
        << "Ctx::desc" << std::endl 
        << props->desc()
        ;

    std::string str = ss.str(); 
    return str ; 
}

std::string Ctx::loglines() const 
{
    std::stringstream ss ; 
    ss << "[Ctx::loglines" << std::endl ; 
    for(int i=0 ; i < int(log.size()) ; i++) ss << log[i] << std::endl ; 
    ss << "]Ctx::loglines" << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}





