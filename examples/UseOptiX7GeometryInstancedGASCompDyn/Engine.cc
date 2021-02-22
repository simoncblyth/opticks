#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <fstream>

#include <optix.h>
#include <optix_stubs.h>
//#include <optix_function_table_definition.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

#include "GAS.h"
#include "IAS.h"
#include "PIP.h"
#include "SBT.h"
#include "Geo.h"
#include "Engine.h"
#include "Binding.h"
#include "Params.h"
#include "SPPM.h"

Engine::Engine(const char* ptx_path_, Params* params_)
    :
    params(params_),
    pip(new PIP(strdup(ptx_path_))),
    sbt(new SBT(pip, params))
{
}

void Engine::setGeo(const Geo* geo_)
{
    sbt->setGeo(geo_); 
}

void Engine::allocOutputBuffer()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_pixels ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_pixels ),
                params->width*params->height*sizeof(uchar4)
                ) );
    params->image = d_pixels ; 
}

void Engine::launch()
{
    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUdeviceptr d_param;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                params, sizeof( Params ),
                cudaMemcpyHostToDevice
                ) );

    OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), params->width, params->height, params->depth ) );
    CUDA_SYNC_CHECK();
}

void Engine::download()
{
    host_pixels.resize(params->width*params->height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( host_pixels.data() ),
                d_pixels,
                params->width*params->height*sizeof(uchar4),
                cudaMemcpyDeviceToHost
    ));
}

void Engine::writePPM(const char* path, bool yflip )
{
    SPPM_write( path, host_pixels.data(), params->width, params->height, yflip );
}

