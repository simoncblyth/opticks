#include "OPTICKS_LOG.hh"

#include <cuda_runtime.h>

#include "SPath.hh"
#include "NP.hh"
#include "QBuf.hh"
#include "QEvent.hh"


// some tests moved down to sysrap/tests/SEventTest.cc

void test_QEvent(const NP* gs)
{
    QEvent* event = new QEvent ; 
    event->setGensteps(gs); 

    unsigned num_photons = event->getNumPhotons() ; 
    assert( num_photons > 0); 

    LOG(info) << event->desc() ; 
    event->seed->download_dump("event->seed", 10); 
    event->checkEvt(); 

    cudaDeviceSynchronize(); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

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
        return 0 ; 
    }


    gs0->dump(); 

    test_QEvent(gs0); 
    return 0 ; 
}

