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

void OPropertyLib::configureSampler(optix::TextureSampler& sampler, optix::Buffer& buffer)
{
    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE ); 
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE );

    RTfiltermode minification = RT_FILTER_LINEAR ;
    RTfiltermode magnification = RT_FILTER_LINEAR ;
    RTfiltermode mipmapping = RT_FILTER_NONE ;

    sampler->setFilteringModes(minification, magnification, mipmapping);

    RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ;
    sampler->setReadMode(readmode);  

    //RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_ARRAY_INDEX ;  // by inspection : zero based array index offset by 0.5
    RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ; // needed by OptiX 400 ? see OptiX_400 pdf p17 
    sampler->setIndexingMode(indexingmode);  


    sampler->setMaxAnisotropy(1.0f);  
    sampler->setMipLevelCount(1u);     
    sampler->setArraySize(1u);        
    // from pdf: OptiX currently supports only a single MIP level and a single element texture array.

    unsigned int texture_array_idx = 0u ;
    unsigned int mip_level = 0u ; 

    sampler->setBuffer(texture_array_idx, mip_level, buffer);
}

/*

OptiX 400

2016-08-10 15:05:35.709 INFO  [1903430] [OContext::launch@214] OContext::launch
entry 0 width 1 height 1 libc++abi.dylib: terminating with uncaught exception
of type optix::Exception: Invalid value (Details: Function "RTresult
_rtContextValidate(RTcontext)" caught exception: Unsupported combination of
texture index, wrap and filter modes:  RT_TEXTURE_INDEX_ARRAY_INDEX,
RT_WRAP_REPEAT, RT_FILTER_LINEAR,
file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Util/TextureDescriptor.cpp,
line: 138) Abort trap: 6

*/






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


