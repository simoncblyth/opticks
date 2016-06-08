#include "OSourceLib.hh"
#include "GSourceLib.hh"

#include "NPY.hpp"
#include "BLog.hh"


void OSourceLib::convert()
{
    LOG(debug) << "OSourceLib::convert" ;
    NPY<float>* buf = m_lib->getBuffer();
    makeSourceTexture(buf);
}

void OSourceLib::makeSourceTexture(NPY<float>* buf)
{
    assert(buf && "OSourceLib::makeSourceTexture NULL buffer, try updating geocache first: ggv -G  ? " );

    unsigned int ni = buf->getShape(0);
    unsigned int nj = buf->getShape(1);
    unsigned int nk = buf->getShape(2);
    assert(ni == 1 && nj == GSourceLib::icdf_length && nk == 1);

    unsigned int nx = nj ;
    unsigned int ny = 1 ;

    LOG(debug) << "OSourceLib::makeSourceTexture "
              << " nx " << nx
              << " ny " << ny  
              ;

    float step = 1.f/float(nx) ;
    optix::float4 domain = optix::make_float4(0.f , 1.f, step, 0.f );
    optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT, nx, ny);

    m_context["source_texture"]->setTextureSampler(tex);
    m_context["source_domain"]->setFloat(domain);
}




