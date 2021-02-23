#include <iostream>
#include <iomanip>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

#include "Frame.h"
#include "Params.h"
#include "SPPM.h"


Frame::Frame(Params* params_)
    :
    params(params_)
{
    init()
}

void Frame::init()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_pixels ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_pixels ),
                params->width*params->height*sizeof(uchar4)
                ) );
    params->image = d_pixels ; 
}

void Frame::download()
{
    pixels.resize(params->width*params->height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( pixels.data() ),
                d_pixels,
                params->width*params->height*sizeof(uchar4),
                cudaMemcpyDeviceToHost
    ));
}

void Frame::writePPM(const char* path, bool yflip )
{
    std::cout << "Frame::writePPM " << path << std::endl ; 
    SPPM_write( path, pixels.data(), params->width, params->height, yflip );
}

