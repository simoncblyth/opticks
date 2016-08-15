
#include "NPY.hpp"
#include "Opticks.hh"
#include "GPropertyLib.hh"
#include "GBndLib.hh"
#include "OBndLib.hh"
#include "PLOG.hh"


OBndLib::OBndLib(optix::Context& ctx, GBndLib* lib)
    : 
    OPropertyLib(ctx),
    m_lib(lib),
    m_texture_debug(false)
{
}

void OBndLib::setTextureDebug(bool dbg)
{
    m_texture_debug = dbg ; 
}


void OBndLib::convert()
{
    LOG(debug) << "OBndLib::convert" ;

    m_lib->createDynamicBuffers();

    NPY<float>* orig = m_lib->getBuffer() ;  // (123, 4, 2, 39, 4)

    NPY<float>* buf = m_texture_debug ? NPY<float>::make_dbg_like(orig, -3) : orig ; 

    assert(buf->hasSameShape(orig));

    makeBoundaryTexture( buf );

    NPY<unsigned int>* obuf = m_lib->getOpticalBuffer() ;  // (123, 4, 4)

    makeBoundaryOptical(obuf);
}


void OBndLib::makeBoundaryTexture(NPY<float>* buf)
{
/*
   b = np.load("/tmp/blyth/opticks/GBndLib/GBndLib.npy")


         #float4
            |     ___ wavelength samples
            |    /
   (123, 4, 2, 39, 4)
    |    |          \___ float4 props        
  #bnd   | 
         |
    omat/osur/isur/imat  

*/

    if(m_texture_debug)
    {
        buf->save("$TMP/OBndLib_makeBoundaryTexture_dbg.npy");
    }

    unsigned int ni = buf->getShape(0);  // (~123) number of boundaries 
    unsigned int nj = buf->getShape(1);  // (4)    number of species : omat/osur/isur/imat 
    unsigned int nk = buf->getShape(2);  // (2)    number of float4 property groups per species 
    unsigned int nl = buf->getShape(3);  // (39)   number of wavelength samples of the property
    unsigned int nm = buf->getShape(4);  // (4)    number of prop within the float4

    assert(ni == m_lib->getNumBnd()) ;
    assert(nj == GPropertyLib::NUM_MATSUR);

    assert(nk == GPropertyLib::NUM_FLOAT4); 
    assert(nl == Opticks::DOMAIN_LENGTH); 
    assert(nm == 4); 

    unsigned int ny = ni*nj*nk ;     // total number of properties from all (two) float4 property groups of all (4) species in all (~123) boundaries 
    unsigned int nx = nl ;           // wavelength samples
   
    LOG(trace) << "OBndLib::makeBoundaryTexture buf " 
              << buf->getShapeString() 
              << " ---> "  
              << " nx " << nx
              << " ny " << ny  
              ;

    
    optix::Buffer optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nx, ny );
    upload(optixBuffer, buf);

    optix::TextureSampler tex = m_context->createTextureSampler();
    configureSampler(tex, optixBuffer);

    unsigned int xmin = 0 ; 
    unsigned int xmax = nx - 1 ; 
    unsigned int ymin = 0 ;
    unsigned int ymax = ny - 1 ; 

    LOG(trace) << "OBndLib::makeBoundaryTexture"
              << " xmin " << xmin 
              << " xmax " << xmax
              << " nx " << nx 
              << " ymin " << ymin 
              << " ymax " << ymax
              << " ny " << ny 
              << " ni " << ni
              << " nj " << nj
              << " nk " << nk
              << " nl " << nl
              << " nm " << nm
              << " ni*nj " << ni*nj
              << " ni*nj*nk " << ni*nj*nk
             ;
 
    optix::uint4 bounds = optix::make_uint4(xmin, xmax, ymin, ymax );

    LOG(trace) << "OBndLib::makeBoundaryTexture bounds (not including the num_float4) " 
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

    LOG(trace) << "OBndLib::makeBoundaryOptical obuf " 
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



