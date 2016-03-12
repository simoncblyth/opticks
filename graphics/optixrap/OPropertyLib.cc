#include "OPropertyLib.hh"

#include "NPY.hpp"
#include "NLog.hpp"


optix::TextureSampler OPropertyLib::makeTexture(NPY<float>* buffer, RTformat format, unsigned int nx, unsigned int ny, bool empty)
{
    unsigned int numBytes = buffer->getNumBytes(0) ;
    optix::Buffer optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT, format, nx, ny );

    LOG(info) << "OPropertyLib::makeTexture"
              << " bufShape " << buffer->getShapeString()
              << " numBytes " << numBytes 
              << " nx " << nx
              << " ny " << ny
              << " empty " << empty 
              ;

    if(!empty)
    {
        memcpy( optixBuffer->map(), buffer->getBytes(), numBytes );
        optixBuffer->unmap(); 
    }

    optix::TextureSampler sampler = m_context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE ); 
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE );

    RTfiltermode minification = RT_FILTER_LINEAR ;
    RTfiltermode magnification = RT_FILTER_LINEAR ;
    RTfiltermode mipmapping = RT_FILTER_NONE ;
    sampler->setFilteringModes(minification, magnification, mipmapping);

    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);  
    sampler->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);  // by inspection : zero based array index offset by 0.5
    sampler->setMaxAnisotropy(1.0f);  
    sampler->setMipLevelCount(1u);     
    sampler->setArraySize(1u);        

    unsigned int texture_array_idx = 0u ;
    unsigned int mip_level = 0u ; 
    sampler->setBuffer(texture_array_idx, mip_level, optixBuffer);

    return sampler ; 
}


