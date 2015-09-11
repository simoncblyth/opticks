
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
    m_num_gensteps = ni ; 
    printf("App::loadGenstep num_gensteps %d\n", m_num_gensteps);
}
void App::initOptiX()
{
    m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );

    m_genstep_buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, m_num_gensteps*6 ); // 6*float4 per genstep 
    m_context["genstep_buffer"]->set( m_genstep_buffer );
    printf("App::initOptiX\n");
}
void App::uploadEvt()
{
    OBuf gs("gs", m_genstep_buffer);

    gs.upload( m_gs );

    // pluck photon counts with atomic view of genstep buffer
    m_num_photons = gs.reduce<unsigned int>(6*4, 3) ;  // stride, offset

    assert(m_num_photons < 3e6 );

    m_photon_buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, m_num_photons*4 ); // 4*float4 per photon

    seedPhotonBuffer(); 

    printf("App::uploadEvt num_photons %u\n", m_num_photons);
}

void App::seedPhotonBuffer()
{



    printf("App::seedPhotonBuffer\n");
}

void App::dumpGensteps()
{
    OBuf gs("gs", m_genstep_buffer) ; 

    gs.dump<float4>("genstep_buffer (f4)", 0, 0, 23 );  // msg, stride, begin, end
    gs.dump<uint4>("genstep_buffer (u4)", 0, 0, 23 );

    gs.dump<unsigned int>("strided 6*4 offset 3 ", 6*4, 3, 23 );  

    printf("App::dumpGensteps\n");
}

void App::dumpPhotons()
{
    OBuf ph("ph", m_photon_buffer) ; 

    ph.dump<float4>("(f4, stride 0)", 0, 0, 10*4 );  // msg, stride, begin, end

    printf("App::dumpPhotons\n");
}

