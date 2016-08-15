#include "NPY.hpp"
#include "OPropertyLib.hh"

#include "PLOG.hh"


OPropertyLib::OPropertyLib(optix::Context& ctx) : m_context(ctx)
{
}

void OPropertyLib::dumpVals( float* vals, unsigned int n)
{
    for(unsigned int i=0 ; i < n ; i++)
    { 
        std::cout << std::setw(10) << vals[i]  ;
        if(i % 16 == 0 ) std::cout << std::endl ; 
    }
}

void OPropertyLib::upload(optix::Buffer& optixBuffer, NPY<float>* buffer)
{
    unsigned int numBytes = buffer->getNumBytes(0) ;
    void* data = buffer->getBytes();
    memcpy( optixBuffer->map(), data, numBytes );
    optixBuffer->unmap(); 
}

//
// NB this requires the memory layout of the optixBuffer needed for the texture of (nx,ny) shape
//    matches that of the NPY<float> buffer
//   
//    this was working for NPY<float> of shape    (128, 4, 39, 4)    
//         going into texture<float4>   nx:39  ny:128*4 boundaries*species
//
//    but for (128, 4, 39, 8)  it did not working 
//    as depends on the 39 being the last dimension of the buffer 
//    excluding the 4 that disappears as payload
//
//    maybe use
//             (128, 4, 39*2, 4 )
//    
//


