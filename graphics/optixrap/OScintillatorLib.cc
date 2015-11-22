#include "OScintillatorLib.hh"
#include "GScintillatorLib.hh"

#include "NPY.hpp"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void OScintillatorLib::convert()
{
    LOG(debug) << "OScintillatorLib::convert" ;
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

    LOG(debug) << "OScintillatorLib::makeReemissionTexture "
              << " nx " << nx
              << " ny " << ny  
              ;

    float step = 1.f/float(nx) ;
    optix::float4 domain = optix::make_float4(0.f , 1.f, step, 0.f );
    optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT, nx, ny);

    m_context["reemission_texture"]->setTextureSampler(tex);
    m_context["reemission_domain"]->setFloat(domain);
}




