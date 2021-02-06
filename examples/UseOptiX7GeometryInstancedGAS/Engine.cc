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
#include "Engine.h"

#include "Binding.h"



static void SPPM_write( const char* filename, const uchar4* image, int width, int height, bool yflip )
{
    FILE * fp; 
    fp = fopen(filename, "wb");

    fprintf(fp, "P6\n%d %d\n%d\n", width, height, 255);

    unsigned size = height*width*3 ; 
    unsigned char* data = new unsigned char[size] ; 

    for( int h=0; h < height ; h++ ) // flip vertically
    {   
        int y = yflip ? height - 1 - h : h ; 

        for( int x=0; x < width ; ++x ) 
        {
            *(data + (y*width+x)*3+0) = image[(h*width+x)].x ;   
            *(data + (y*width+x)*3+1) = image[(h*width+x)].y ;   
            *(data + (y*width+x)*3+2) = image[(h*width+x)].z ;   
        }
    }   
    fwrite(data, sizeof(unsigned char)*size, 1, fp);
    fclose(fp);  
    std::cout << "Wrote file (uchar4) " << filename << std::endl  ;
    delete[] data;
}

OptixDeviceContext Engine::context = nullptr ;

void Engine::context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)  // static 
{
    std::cerr 
        << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
        << message << "\n";
}


Engine::Engine(const char* ptx_path_)
    :
    rc(preinit()),
    bb({-1.5,-1.5,-1.5,1.5,1.5,1.5}),
    gas(GAS::Build(bb)),
    ias(gas.handle),
    pip(strdup(ptx_path_))
{
    init();  
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

void Engine::init()
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

    //params.handle       = gas.gas_handle;  // OK
    //params.handle       = ias.gas_handle;  // OK 
    params.handle       = ias.ias_handle;    // now OK when set 

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


