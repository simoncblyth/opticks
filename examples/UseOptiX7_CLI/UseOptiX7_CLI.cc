/**
UseOptiX7_CLI.cc
==================

~/o/examples/UseOptiX7_CLI/go.sh 
~/o/examples/UseOptiX7_CLI/UseOptiX7_CLI.cc

**/
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#define _xstr(s) _str(s)
#define _str(s) #s

#include "CUDA_CHECK.h"
#include "OPTIX_CHECK.h"


void Ctx__log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)  // static 
{   
    std::stringstream ss ; 
    ss << "[" << std::setw(2) << level << "][" << std::setw( 12 ) << tag << "]: " << message ;
    std::string line = ss.str() ;
    std::cout << line ;
}


int main()
{
    const char* vers = _xstr(OPTIX_VERSION) ; 
    std::cout << "OPTIX_VERSION : " << vers << std::endl ; 


    // Initialize CUDA and create OptiX context

    OptixDeviceContext context = nullptr;
    {
        CUDA_CHECK(cudaFree( 0 ) );  

        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK( optixInit() );
        OptixDeviceContextOptions options = {}; 
        options.logCallbackFunction       = &Ctx__log_cb;
        options.logCallbackLevel          = 4;
        //options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL ; 

        OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
    }




    return 0 ; 
}
