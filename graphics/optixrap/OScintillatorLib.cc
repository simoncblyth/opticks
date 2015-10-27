#include "OScintillatorLib.hh"
#include "GScintillatorLib.hh"

#include "NPY.hpp"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void OScintillatorLib::convert()
{
    LOG(info) << "OScintillatorLib::convert" ;
    NPY<float>* buf = m_lib->getBuffer();
    makeReemissionTexture(buf);
}

void OScintillatorLib::makeReemissionTexture(NPY<float>* buf)
{
    unsigned int ni = buf->getShape(0);
    unsigned int nj = buf->getShape(1);
    unsigned int nk = buf->getShape(2);
    assert(ni == 1 && nj == 4096 && nk == 1);

    unsigned int nx = nj ;
    unsigned int ny = 1 ;

    LOG(info) << "OScintillatorLib::makeReemissionTexture "
              << " nx " << nx
              << " ny " << ny  
              ;

    float step = 1.f/float(nx) ;
    optix::float4 domain = optix::make_float4(0.f , 1.f, step, 0.f );
    optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT, nx, ny);

    m_context["reemission_texture"]->setTextureSampler(tex);
    m_context["reemission_domain"]->setFloat(domain);
}


optix::TextureSampler OScintillatorLib::makeTexture(NPY<float>* buffer, RTformat format, unsigned int nx, unsigned int ny)
{
    unsigned int numBytes = buffer->getNumBytes(0) ;
    optix::Buffer optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT, format, nx, ny );
    memcpy( optixBuffer->map(), buffer->getBytes(), numBytes );
    optixBuffer->unmap(); 

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



