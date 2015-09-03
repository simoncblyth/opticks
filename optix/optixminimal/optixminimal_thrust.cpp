#include <string>
#include <iostream>
#include <cstdlib>
#include <cassert>

#include "cuda_runtime_api.h"
#include <optixu/optixpp_namespace.h>
#include "thrust_simple.hh"


#define CUDA_SAFE_CALL( call) do {                                         \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                      \
    } } while (0)



std::string ptxpath(const char* name){
    char path[128] ; 
    snprintf(path, 128, "%s/%s", getenv("PTXDIR"), name );
    return path ;  
}

enum { raygen_entry, num_entry } ;
unsigned int width  = 512 ; 
unsigned int height = 512 ; 

int main( int argc, char** argv )
{
    optix::Context context = optix::Context::create();
    context->setPrintEnabled(true);
    context->setPrintBufferSize(8192);
    context->setStackSize( 2180 );
    context->setEntryPointCount(num_entry);

    //unsigned int type = RT_BUFFER_OUTPUT ; // means is written on the GPU
    unsigned int type = RT_BUFFER_INPUT_OUTPUT ; 
    optix::Buffer buffer = context->createBuffer(type, RT_FORMAT_UNSIGNED_BYTE4, width, height );
    context["output_buffer"]->set( buffer );

    optix::Program raygen = context->createProgramFromPTXFile( ptxpath("minimal.ptx"), "minimal" );
    context->setRayGenerationProgram( raygen_entry, raygen );

    context->validate();
    context->compile();
    context->launch(0,0);
    context->launch(raygen_entry, width, height);

    // bring to host and dump
    unsigned char* ptr = (unsigned char*)buffer->map();
    for(unsigned int i=0 ; i < width*height*4 ; i++){
        unsigned char v = *(ptr + i );
        assert( v == 128 );
        //std::cout << int(v) << ( i % width == 0 ? "\n" : " " ) ; 
    }

    buffer->unmap();
    buffer->markDirty();

    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

    // try on device access with thrust    
    bool thrust = true ; 
    if(thrust)
    {
        unsigned int device_number = 0 ; 
        typedef unsigned char T ; 

        optix::Buffer buf = context["output_buffer"]->getBuffer();
        CUdeviceptr cu_ptr = buf->getDevicePointer(device_number);
        T* d_ptr = reinterpret_cast<T*>(cu_ptr)  ;

        printf(" cu_ptr %llu d_ptr %p \n", cu_ptr, static_cast<void*>(d_ptr) );

        unsigned int size = width*height*4 ;
        for(unsigned int i=0 ; i < 256 ; i++ )
        {
            T val = (unsigned char)i ; 
            T num = thrust_simple_count<T>( d_ptr, size, val ); 
            if(num > 0 ) printf("thrust val %d num %d  \n", val, num );
        }
    }






    return 0 ; 
}

