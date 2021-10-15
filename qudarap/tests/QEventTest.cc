#include "OPTICKS_LOG.hh"

#include <cuda_runtime.h>
#include "QBuf.hh"
#include "QEvent.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    /*
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    unsigned x_total = 0 ; 
    for(unsigned i=0 ; i < photon_counts_per_genstep.size() ; i++) x_total += photon_counts_per_genstep[i] ; 
    const NP* gs = QEvent::MakeCountGensteps(photon_counts_per_genstep) ; 
    */

    float4 ce = make_float4( 0.f, 0.f, 0.f, 100.f ); 
    uint4 cegs = make_uint4( 3, 0, 3, 100 ); 
    float gridscale = 1.f ; 

    const NP* gs = QEvent::MakeCenterExtentGensteps(ce, cegs, gridscale); 

    QEvent* event = new QEvent ; 
    event->setGensteps(gs); 

    unsigned num_photons = event->getNumPhotons() ; 
    assert( num_photons > 0); 

    //assert( event->getNumPhotons() == x_total ); 

    LOG(info) << event->desc() ; 
    event->seed->download_dump("event->seed", 10); 

    event->checkEvt(); 

    cudaDeviceSynchronize(); 

    return 0 ; 
}

