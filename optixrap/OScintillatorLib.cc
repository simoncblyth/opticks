#include "NPY.hpp"
#include "GScintillatorLib.hh"
#include "OConfig.hh"
#include "OScintillatorLib.hh"

#include "PLOG.hh"
// trace/debug/info/warning/error/fatal

OScintillatorLib::OScintillatorLib(optix::Context& ctx, GScintillatorLib* lib)
           : 
           OPropertyLib(ctx),
           m_lib(lib)
{
}


void OScintillatorLib::convert(const char* slice)
{
    NPY<float>* buf = m_lib->getBuffer();
    unsigned ni = buf->getShape(0) ; 
    LOG(trace) << "OScintillatorLib::convert" 
               << " from " << buf->getShapeString() 
               << " ni " << ni 
               ;


    if( ni > 1 && slice )
    { 
        NPY<float>* slice_buf = buf->make_slice(slice) ;

        LOG(info) << "OScintillatorLib::convert" 
                  << " sliced buffer with " << slice
                  << " from " << buf->getShapeString()
                  << " to " << slice_buf->getShapeString()
                  ;
 
        makeReemissionTexture(slice_buf);
    }
    else
    {
        makeReemissionTexture(buf);
    }

    LOG(trace) << "OScintillatorLib::convert DONE" ;
}

void OScintillatorLib::makeReemissionTexture(NPY<float>* buf)
{
    if(!buf)
    {
       LOG(fatal) << "OScintillatorLib::makeReemissionTexture MISSING BUFFER " ;
       LOG(fatal) << " you probably need to populate the geocache for the current geometry selection " ;

    } 
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
    {
        LOG(error) << "OScintillatorLib::makeReemissionTexture no scintillators, skipping " ;
        return ;   
    }
 
    optix::Buffer optixBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, nx, ny );

    upload(optixBuffer, buf);

    optix::TextureSampler tex = m_context->createTextureSampler();
    OConfig::configureSampler(tex, optixBuffer);

    m_context["reemission_texture"]->setTextureSampler(tex);
    m_context["reemission_domain"]->setFloat(domain);

    LOG(trace) << "OScintillatorLib::makeReemissionTexture DONE " ; 
}




