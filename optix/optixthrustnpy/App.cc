
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
#include "OBufPair.hh"
#include "NPY.hpp"



void App::loadGenstep()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1", "dayabay");
    npy->Summary();
    unsigned int ni = npy->getShape(0) ;
    unsigned int nj = npy->getShape(1) ; 
    unsigned int nk = npy->getShape(2) ;
    assert(nk == 4 && nj == 6); 
    m_gs_npy = npy ; 
    m_num_gensteps = ni ; 
    printf("App::loadGenstep num_gensteps %d\n", m_num_gensteps);
}
void App::initOptiX()
{
    m_context = optix::Context::create();
    m_context->setPrintEnabled(true);
    m_context->setPrintBufferSize(8192);
    m_context->setStackSize( 2180 );

    printf("App::initOptiX\n");
}
void App::uploadEvt()
{
    m_genstep_buffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, m_num_gensteps*6 ); // 6*float4 per genstep 
    m_context["genstep_buffer"]->set( m_genstep_buffer );

    OBuf gs("gs", m_genstep_buffer);
    gs.upload( m_gs_npy );

    // pluck photon counts with atomic view of genstep buffer
    m_num_photons = gs.reduce<unsigned int>(6*4, 3) ;  // stride, offset
    assert(m_num_photons < 3e6 );

    m_photon_buffer = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, m_num_photons*4 ); // 4*float4 per photon
    m_context["photon_buffer"]->set( m_photon_buffer );
    OBuf ph("ph", m_photon_buffer) ; 

    OBufPair<unsigned int> bp(gs.slice(6*4,3,0), ph.slice(4*4,0,0));
    bp.seedDestination();  

    // **OBufPair::seedDestination** 
    //
    //      Distributes unsigned int genstep indices 0:m_num_gensteps-1 into the first 
    //      4 bytes of the 4*float4 photon record in the photon buffer 
    //      using the number of photons per genstep obtained from the genstep buffer 
    //
    //      Note that this is done almost entirely on the GPU, only the num_photons reduction
    //      needs to come back to CPU in order to allocate an appropriately sized OptiX photon 
    //      buffer on GPU.
    //
    //      This per-photon genstep index is used by OptiX photon propagation 
    //      program cu/generate.cu to access the appropriate values from the genstep buffer
    //

    printf("App::uploadEvt num_photons %u\n", m_num_photons);
}

void App::downloadEvt()
{
    NPY<float>* npy = NPY<float>::make(m_num_photons, 4,4);  // host allocation

    OBuf ph("ph", m_photon_buffer) ; 
    ph.download(npy);

    npy->save("/tmp/ph.npy");     

/*
    In [9]: p = np.load("/tmp/ph.npy")

    In [14]: p.shape
    Out[14]: (612841, 4, 4)

    In [13]: p.view(np.uint32)[:,0,0]
    Out[13]: array([   0,    0,    0, ..., 7835, 7835, 7835], dtype=uint32)

    In [18]: np.all( np.arange(0,7836) == np.unique(p.view(np.uint32)[:,0,0]) )
    Out[18]: True

*/
    printf("App::downloadEvt num_photons %u\n", m_num_photons);
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

