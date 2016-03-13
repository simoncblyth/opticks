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

    NPY<float>* buf = m_lib->getBuffer() ;
    buf->save("/tmp/OBndLib_convert_bndbuf.npy");

    makeBoundaryTexture( buf );

    makeBoundaryOptical(m_lib->getOpticalBuffer());
}


void OBndLib::makeBoundaryTexture(NPY<float>* buf)
{
    //  eg (123, 4, 39, 4)   boundary, imat-omat-isur-osur, wavelength-samples, 4-props

    unsigned int ni = buf->getShape(0);  // number of boundaries
    unsigned int nj = buf->getShape(1);  // number of species:4  omat/osur/isur/imat 
    unsigned int nk = buf->getShape(2);  
    unsigned int nl = buf->getShape(3); 
    unsigned int nm = buf->getShape(4);  

    assert(ni == m_lib->getNumBnd()) ;
    assert(nj == GPropertyLib::NUM_MATSUR);

    unsigned int nx ;
    unsigned int ny ;   

    if(nm == 0)
    {
        assert(nk == Opticks::DOMAIN_LENGTH); 
        assert(nl == GPropertyLib::NUM_PROP );
        assert(nl == 4 || nl == 8);
        unsigned int n_float4 = nl/4 ; 
        assert( n_float4 == GPropertyLib::NUM_FLOAT4 );

        ny = ni*nj*n_float4 ;   //
        nx = nk ;               // wavelength samples

        assert(0 && "old style buffer not supported");
    }
    else
    {
        assert(nk == GPropertyLib::NUM_FLOAT4); 
        assert(nl == Opticks::DOMAIN_LENGTH); 
        assert(nm == 4); 

        ny = ni*nj*nk ;
        nx = nl ;           // wavelength samples
    }

   
    LOG(info) << "OBndLib::makeBoundaryTexture buf " 
              << buf->getShapeString() 
              << " ---> "  
              << " nx " << nx
              << " ny " << ny  
              ;

    optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT4, nx, ny);

    unsigned int wmin = 0 ; 
    unsigned int wmax = nx - 1 ; 
    unsigned int lmin = m_lib->getLineMin() ;
    unsigned int lmax = m_lib->getLineMax() ; 

    // huh factor of 2 somewhere ???  nope payload details are beneath texture line level

    LOG(info) << "OBndLib::makeBoundaryTexture"
              << " lmin " << lmin 
              << " lmax " << lmax
              << " ni " << ni
              << " nj " << nj
              << " nk " << nk
              << " nl " << nl
              << " nm " << nm
             ;
 
    assert(lmin == 0 && lmax == ni*nj - 1);

    optix::uint4 bounds = optix::make_uint4(wmin, wmax, lmin, lmax );

    LOG(debug) << "OBndLib::makeBoundaryTexture bounds (not including the num_float4) " 
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
    unsigned int numBnd = numBytes/(GPropertyLib::NUM_MATSUR*4*sizeof(unsigned int)) ;  // this 4 is not NUM_PROP
    unsigned int nx = numBnd*GPropertyLib::NUM_MATSUR ;

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



