#include "OBndLib.hh"
#include "GBndLib.hh"
#include "GPropertyLib.hh"
#include "NPY.hpp"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void OBndLib::convert()
{
    LOG(info) << "OBndLib::convert" ;

    m_lib->createDynamicBuffer();

    makeBoundaryTexture(m_lib->getBuffer());

    // hmm the index and optical buffer are not the same.. optical has the GOpticalSurface content...
    //makeBoundaryIndex(m_lib->getIndexBuffer());
}

void OBndLib::makeBoundaryTexture(NPY<float>* buf)
{
    //  eg (123, 4, 39, 4)   boundary, imat-omat-isur-osur, wavelength-samples, 4-props

    unsigned int ni = buf->getShape(0);
    unsigned int nj = buf->getShape(1);
    unsigned int nk = buf->getShape(2);
    unsigned int nl = buf->getShape(3);

    assert(ni == m_lib->getNumBnd()) ;
    assert(nj == GPropertyLib::NUM_QUAD && nk == GPropertyLib::DOMAIN_LENGTH && nl == GPropertyLib::NUM_PROP );

    unsigned int nx = nk ;
    unsigned int ny = ni*nj ;   // not nl as using float4

    LOG(info) << "OBndLib::makeBoundaryTexture buf " 
              << buf->getShapeString() 
              << " ---> "  
              << " nx " << nx
              << " ny " << ny  
              ;

    optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT4, nx, ny);

    unsigned int wmin = 0 ; 
    unsigned int wmax = nk - 1 ; 
    unsigned int lmin = m_lib->getLineMin() ;
    unsigned int lmax = m_lib->getLineMax() ;
    assert(lmin == 0 && lmax == ni*nj - 1);

    optix::uint4 bounds = optix::make_uint4(wmin, wmax, lmin, lmax );

    LOG(info) << "OBndLib::makeBoundaryTexture bounds " 
              << " x " << bounds.x 
              << " y " << bounds.y
              << " z " << bounds.z 
              << " w " << bounds.w 
              ;


    float domain_range = (GPropertyLib::DOMAIN_HIGH - GPropertyLib::DOMAIN_LOW); 
    optix::float4 domain = optix::make_float4(GPropertyLib::DOMAIN_LOW, GPropertyLib::DOMAIN_HIGH, GPropertyLib::DOMAIN_STEP, domain_range); 

    // only endpoints used for sampling, not the step 
    optix::float4 domain_reciprocal = optix::make_float4(1.f/GPropertyLib::DOMAIN_LOW, 1.f/GPropertyLib::DOMAIN_HIGH, 0.f, 0.f); // not flipping order 

    // formerly(with OBoundaryLib)  prefixed wavelength_ 
    m_context["boundary_texture"]->setTextureSampler(tex);
    m_context["boundary_domain"]->setFloat(domain); 
    m_context["boundary_domain_reciprocal"]->setFloat(domain_reciprocal); 
    m_context["boundary_bounds"]->setUint(bounds); 
}


/*
void OBndLib::makeBoundaryIndex(NPY<unsigned int>* buf)
{
    unsigned int numBytes = buf->getNumBytes(0) ;
    unsigned int numBnd = numBytes/(NUM_QUAD*NUM_PROP*sizeof(unsigned int)) ;
    assert( buf->getShape(0) == numBnd );


    optix::Buffer optical_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, numBoundaries*6 );
    memcpy( optical_buffer->map(), obuf->getBytes(), obuf->getNumBytes() );
    optical_buffer->unmap();
    //optix::Buffer optical_buffer = createInputBuffer<unsigned int>( obuf, RT_FORMAT_UNSIGNED_INT4, 4);
    m_context["optical_buffer"]->setBuffer(optical_buffer);
}
*/




