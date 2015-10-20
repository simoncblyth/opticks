#include "OBoundaryLib.hh"

#include "GGeo.hh"
#include "GBoundaryLib.hh"

// npy-
#include "stringutil.hpp"

//#include "OConfig.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void OBoundaryLib::convert()
{
    convertBoundaryProperties(m_boundarylib);
    LOG(info) << "OBoundaryLib::convert"
              ;
}


optix::TextureSampler OBoundaryLib::makeWavelengthSampler(GBuffer* buffer)
{
   // handles different numbers of substances, but uses static domain length
    unsigned int domainLength = GBoundaryLib::DOMAIN_LENGTH ;
    unsigned int numElementsTotal = buffer->getNumElementsTotal();
    assert( numElementsTotal % domainLength == 0 );

    unsigned int nx = domainLength ;
    unsigned int ny = numElementsTotal / domainLength ;

    LOG(info) << "OBoundaryLib::makeWavelengthSampler "
              << " numElementsTotal " << numElementsTotal  
              << " (nx)domainLength " << domainLength 
              << " ny (props*subs)  " << ny 
              << " ny/(6*4) " << ny/(6*4) ; 

    optix::TextureSampler sampler = makeSampler(buffer, RT_FORMAT_FLOAT4, nx, ny);
    return sampler ; 
}

optix::TextureSampler OBoundaryLib::makeReemissionSampler(GBuffer* buffer)
{
    unsigned int domainLength = buffer->getNumElementsTotal();
    unsigned int nx = domainLength ;
    unsigned int ny = 1 ;

    LOG(info) << "OBoundaryLib::makeReemissionSampler "
              << " (nx)domainLength " << domainLength 
              << " ny " << ny  ;

    optix::TextureSampler sampler = makeSampler(buffer, RT_FORMAT_FLOAT, nx, ny);
    return sampler ; 
}

optix::float4 OBoundaryLib::getDomain()
{
    float domain_range = (GBoundaryLib::DOMAIN_HIGH - GBoundaryLib::DOMAIN_LOW); 
    return optix::make_float4(GBoundaryLib::DOMAIN_LOW, GBoundaryLib::DOMAIN_HIGH, GBoundaryLib::DOMAIN_STEP, domain_range); 
}

optix::float4 OBoundaryLib::getDomainReciprocal()
{
    // only endpoints used for sampling, not the step 
    return optix::make_float4(1./GBoundaryLib::DOMAIN_LOW, 1./GBoundaryLib::DOMAIN_HIGH, 0.f, 0.f); // not flipping order 
}

void OBoundaryLib::convertBoundaryProperties(GBoundaryLib* blib)
{
    GBuffer* wavelengthBuffer = blib->getWavelengthBuffer();
    optix::TextureSampler wavelengthSampler = makeWavelengthSampler(wavelengthBuffer);

    optix::float4 wavelengthDomain = getDomain();
    optix::float4 wavelengthDomainReciprocal = getDomainReciprocal();
    optix::uint4 wavelengthBounds = optix::make_uint4(0, GBoundaryLib::DOMAIN_LENGTH - 1, blib->getLineMin(), blib->getLineMax() );

    LOG(info) << "OBoundaryLib::convertBoundaryProperties wavelengthBounds " 
              << " x " << wavelengthBounds.x 
              << " y " << wavelengthBounds.y
              << " z " << wavelengthBounds.z 
              << " w " << wavelengthBounds.w 
              ;

    m_context["wavelength_texture"]->setTextureSampler(wavelengthSampler);
    m_context["wavelength_domain"]->setFloat(wavelengthDomain); 
    m_context["wavelength_domain_reciprocal"]->setFloat(wavelengthDomainReciprocal); 
    m_context["wavelength_bounds"]->setUint(wavelengthBounds); 

    GBuffer* obuf = blib->getOpticalBuffer();

    unsigned int numBoundaries = obuf->getNumBytes()/(4*6*sizeof(unsigned int)) ;
    optix::Buffer optical_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, numBoundaries*6 );
    memcpy( optical_buffer->map(), obuf->getPointer(), obuf->getNumBytes() );
    optical_buffer->unmap();
    //optix::Buffer optical_buffer = createInputBuffer<unsigned int>( obuf, RT_FORMAT_UNSIGNED_INT4, 4);
    m_context["optical_buffer"]->setBuffer(optical_buffer);


    GBuffer* reemissionBuffer = blib->getReemissionBuffer();
    float reemissionStep = 1.f/reemissionBuffer->getNumElementsTotal() ; 
    optix::float4 reemissionDomain = optix::make_float4(0.f , 1.f, reemissionStep, 0.f );
    optix::TextureSampler reemissionSampler = makeReemissionSampler(reemissionBuffer);

    m_context["reemission_texture"]->setTextureSampler(reemissionSampler);
    m_context["reemission_domain"]->setFloat(reemissionDomain);
}



optix::TextureSampler OBoundaryLib::makeSampler(GBuffer* buffer, RTformat format, unsigned int nx, unsigned int ny)
{
    optix::Buffer optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT, format, nx, ny );
    memcpy( optixBuffer->map(), buffer->getPointer(), buffer->getNumBytes() );
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







/*
optix::TextureSampler OGeo::makeColorSampler(unsigned int nx)
{
    unsigned char* colors = make_uchar4_colors(nx);
    unsigned int numBytes = nx*sizeof(unsigned char)*4 ; 

    optix::Buffer optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4, nx );
    memcpy( optixBuffer->map(), colors, numBytes );
    optixBuffer->unmap(); 

    delete colors ; 

    optix::TextureSampler sampler = m_context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE ); 

    RTfiltermode minification = RT_FILTER_LINEAR ;
    RTfiltermode magnification = RT_FILTER_LINEAR ;
    RTfiltermode mipmapping = RT_FILTER_NONE ;
    sampler->setFilteringModes(minification, magnification, mipmapping);

    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);  
    sampler->setIndexingMode(RT_TEXTURE_INDEX_ARRAY_INDEX);  // by inspection : zero based array index offset by 0.5
    sampler->setMaxAnisotropy(1.0f);  
    sampler->setMipLevelCount(1u);     
    sampler->setArraySize(1u);          //  num_textures_in_array

    unsigned int texture_array_idx = 0u ;
    unsigned int mip_level = 0u ; 
    sampler->setBuffer(texture_array_idx, mip_level, optixBuffer);

    return sampler ; 
}
*/


