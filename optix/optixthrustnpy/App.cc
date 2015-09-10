
#include "assert.h"
#include "stdio.h"

#include <vector_types.h>

// *vector_types.h* has to come before OptiX headers to 
// avoid linkage problems due to some tickery 
// with optix::float4 and float4 matchup 
//
//     namespace optix {
//        #include <vector_types.h>
//     }
//
// see https://devtalk.nvidia.com/default/topic/574078/?comment=3896854

#include "App.hh"
#include "OBuf.hh"
#include "NPY.hpp"



void App::loadGenstep()
{
    NPY<float>* gs = NPY<float>::load("cerenkov","1", "dayabay");
    gs->Summary();
    unsigned int ni = gs->getShape(0) ;
    unsigned int nj = gs->getShape(1) ; 
    unsigned int nk = gs->getShape(2) ;
    assert(nk == 4 && nj == 6); 
    m_gs = gs ; 
    m_gs_size = ni*nj  ; 
    printf("App::loadGenstep\n");
}
void App::initOptiX()
{
    m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );

    m_genstep_buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, m_gs_size );
    m_context["genstep_buffer"]->set( m_genstep_buffer );
    printf("App::initOptiX\n");
}
void App::uploadGenstep()
{
    // TODO: tuck away this uploading into reusable place, maybe in optixrap-

    void* data = m_gs->getBytes() ;
    assert(data);
    unsigned int numBytes = m_gs->getNumBytes(0);
    printf("App::uploadGenstep nbytes %u \n", numBytes);
    memcpy( m_genstep_buffer->map(), data, numBytes );
    m_genstep_buffer->unmap();

    printf("App::uploadGenstep\n");
}
void App::checkGenstep()
{
    //dumpBuffer<float4>("genstep_buffer", m_genstep_buffer, 0, 23 );

    OBuf<float4> gsf4(m_genstep_buffer) ;
    gsf4.dump("genstep_buffer (f4)", 0, 23 );

    OBuf<uint4> gsu4(m_genstep_buffer) ;
    gsu4.dump("genstep_buffer (u4)", 0, 23 );

    printf("App::checkGenstep\n");
}


