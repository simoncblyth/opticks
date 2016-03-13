#include "OBndLib.hh"
#include "GBndLib.hh"


#include "Opticks.hh"
#include "GPropertyLib.hh"

#include "NPY.hpp"
#include "NLog.hpp"


void OBndLib::convert()
{
    LOG(debug) << "OBndLib::convert" ;

    m_lib->createDynamicBuffers();

    makeBoundaryTexture(m_lib->getBuffer());

    makeBoundaryOptical(m_lib->getOpticalBuffer());
}


void OBndLib::makeBoundaryTexture(NPY<float>* buf)
{
    //  eg (123, 4, 39, 4)   boundary, imat-omat-isur-osur, wavelength-samples, 4-props

    unsigned int ni = buf->getShape(0);  // number of boundaries
    unsigned int nj = buf->getShape(1);  // number of species:4  omat/osur/isur/imat 
    unsigned int nk = buf->getShape(2);  // number of wavelength samples 
    unsigned int nl = buf->getShape(3);  // number of properties

    assert(ni == m_lib->getNumBnd()) ;
    assert(nj == GPropertyLib::NUM_QUAD && nk == Opticks::DOMAIN_LENGTH && nl == GPropertyLib::NUM_PROP );

    assert(nl == 4 || nl == 8);
    unsigned int n_float4 = nl/4 ; 

    unsigned int nx = nk ;               // wavelength samples
    unsigned int ny = ni*nj*n_float4 ;   //
   
    LOG(info) << "OBndLib::makeBoundaryTexture buf " 
              << buf->getShapeString() 
              << " ---> "  
              << " nx " << nx
              << " ny " << ny  
              << " n_float4 " << n_float4  
              ;

    optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT4, nx, ny);

    unsigned int wmin = 0 ; 
    unsigned int wmax = nk - 1 ; 
    unsigned int lmin = m_lib->getLineMin() ;
    unsigned int lmax = m_lib->getLineMax() ;   // huh factor of 2 somewhere ???

    LOG(info) << "OBndLib::makeBoundaryTexture"
              << " lmin " << lmin 
              << " lmax " << lmax
              << " ni " << ni
              << " nj " << nj
             ;
 
    assert(lmin == 0 && lmax == ni*nj - 1);

    optix::uint4 bounds = optix::make_uint4(wmin, wmax, lmin, lmax );

    LOG(debug) << "OBndLib::makeBoundaryTexture bounds " 
              << " x " << bounds.x 
              << " y " << bounds.y
              << " z " << bounds.z 
              << " w " << bounds.w 
              ;

    glm::vec4 dom = Opticks::getDefaultDomainSpec() ;
    glm::vec4 rdom( 1.f/dom.x, 1.f/dom.y , 0.f, 0.f ); // not flipping order, only endpoints used for sampling, not the step 

    m_context["boundary_texture"]->setTextureSampler(tex);
    m_context["boundary_domain"]->setFloat(dom.x, dom.y, dom.z, dom.w); 
    m_context["boundary_domain_reciprocal"]->setFloat(rdom.x, rdom.y, rdom.z, rdom.w); 
    m_context["boundary_bounds"]->setUint(bounds); 
}


void OBndLib::makeBoundaryOptical(NPY<unsigned int>* obuf)
{
    unsigned int numBytes = obuf->getNumBytes(0) ;
    unsigned int numBnd = numBytes/(GPropertyLib::NUM_QUAD*4*sizeof(unsigned int)) ;  // this 4 is not NUM_PROP
    unsigned int nx = numBnd*GPropertyLib::NUM_QUAD ;

    LOG(info) << "OBndLib::makeBoundaryOptical obuf " 
              << obuf->getShapeString() 
              << " numBnd " << numBnd 
              << " numBytes " << numBytes 
              << " nx " << nx
              ;

    assert( obuf->getShape(0) == numBnd );

    optix::Buffer optical_buffer = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT4, nx );
    memcpy( optical_buffer->map(), obuf->getBytes(), numBytes );
    optical_buffer->unmap();

    m_context["optical_buffer"]->setBuffer(optical_buffer);
}



