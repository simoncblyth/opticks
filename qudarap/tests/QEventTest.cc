#include "OPTICKS_LOG.hh"

#include <cuda_runtime.h>

#include "SPath.hh"
#include "NP.hh"
#include "QBuf.hh"

#include "scuda.h"
#include "squad.h"

#include "SEvent.hh"
#include "QEvent.hh"

// some tests moved down to sysrap/tests/SEventTest.cc

void test_setGensteps_0()
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    //unsigned x_total = SEvent::SumCounts( photon_counts_per_genstep ) ; 
    const NP* gs = SEvent::MakeCountGensteps(photon_counts_per_genstep) ; 

    QEvent* event = new QEvent ; 
    LOG(info) << " event.desc " << event->desc() ; 
    event->setGensteps(gs); 

    //unsigned count_genstep_photons  = event->count_genstep_photons(); 
    //LOG(info) << " count_genstep_photons " << count_genstep_photons << " x_total " << x_total ; 
    //assert( x_total == count_genstep_photons ); 
}

void test_setGensteps_1(const NP* gs)
{
    QEvent* event = new QEvent ; 
    event->setGensteps(gs); 

    unsigned num_photons = event->getNumPhotons() ; 
    assert( num_photons > 0); 

    LOG(info) << event->desc() ; 
    //event->seed->download_dump("event->seed", 10); 
    event->checkEvt(); 
}

void test_setGensteps_1()
{
    const char* path_ = "$TMP/sysrap/SEventTest/cegs.npy" ;
    const char* path = SPath::Resolve(path_, 0); 
    const NP* gs0 = NP::Load(path); 

    if( gs0 == nullptr )
    {
        LOG(fatal) 
            << "failed to load from"
            << " path_ " << path_
            << " path " << path 
            ;
        assert(0); 
    }

    gs0->dump(); 
    test_setGensteps_1(gs0); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_setGensteps_0(); 

    cudaDeviceSynchronize(); 
    return 0 ; 
}

