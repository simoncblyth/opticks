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
    assert(buf) ;  

    unsigned int ni = buf->getShape(0);
    unsigned int nj = buf->getShape(1);
    unsigned int nk = buf->getShape(2);

    bool empty = ni == 0 ;
     
    unsigned int nx = 4096 ; 
    unsigned int ny = 1 ; 
    float step = 1.f/float(nx) ;
    optix::float4 domain = optix::make_float4(0.f , 1.f, step, 0.f );

    LOG(info) << "OScintillatorLib::makeReemissionTexture "
              << " nx " << nx
              << " ny " << ny  
              << " ni " << ni  
              << " nj " << nj  
              << " nk " << nk
              << " step " << step
              << " empty " << empty
              ;

    if(empty)
        LOG(warning) << "OScintillatorLib::makeReemissionTexture no scintillators, creating empty texture " ;

    optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT, nx, ny, empty);
    m_context["reemission_texture"]->setTextureSampler(tex);
    m_context["reemission_domain"]->setFloat(domain);
}




