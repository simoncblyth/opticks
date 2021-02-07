#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <fstream>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

#include "GAS.h"
#include "IAS.h"
#include "Geo.h"
#include "Engine.h"

#include "Binding.h"
#include "SPPM.h"

OptixDeviceContext Engine::context = nullptr ;

void Engine::context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)  // static 
{
    std::cerr 
        << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
        << message << "\n";
}

int Engine::preinit()
{
    CUDA_CHECK( cudaFree( 0 ) ); // Initialize CUDA

    CUcontext cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &Engine::context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );

    return 0 ; 
}

Engine::Engine(const char* ptx_path_)
    :
    rc(preinit()),
    geo(new Geo),
    pip(strdup(ptx_path_))
{
}


void Engine::setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_)
{
    pip.setView(eye_, U_, V_, W_); 
}

void Engine::setSize(unsigned width_, unsigned height_)
{
    width = width_ ; 
    height = height_ ; 
}

void Engine::allocOutputBuffer()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( device_pixels ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &device_pixels ),
                width*height*sizeof(uchar4)
                ) );
}


void Engine::launch()
{
    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );

    Params params;
    params.image        = device_pixels ;
    params.image_width  = width;
    params.image_height = height;
    params.origin_x     = width / 2;
    params.origin_y     = height / 2;
    params.handle       = geo->getTop(); 

    CUdeviceptr d_param;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                &params, sizeof( params ),
                cudaMemcpyHostToDevice
                ) );

    OPTIX_CHECK( optixLaunch( pip.pipeline, stream, d_param, sizeof( Params ), &pip.sbt, width, height, /*depth=*/1 ) );
    CUDA_SYNC_CHECK();
}

void Engine::download()
{
    host_pixels.resize(width*height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( host_pixels.data() ),
                device_pixels,
                width*height*sizeof(uchar4),
                cudaMemcpyDeviceToHost
    ));
}

void Engine::writePPM(const char* path)
{
    bool yflip = true ;  
    SPPM_write( path,  host_pixels.data() , width, height, yflip );
}


