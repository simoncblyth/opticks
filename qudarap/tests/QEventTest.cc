#include "OPTICKS_LOG.hh"

#include <cuda_runtime.h>

#include "SPath.hh"
#include "QBuf.hh"
#include "QEvent.hh"

const char* BASE = "$TMP/qudarap/QEventTest" ; 


const NP* test_MakeCountGensteps()
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    unsigned x_total = 0 ; 
    for(unsigned i=0 ; i < photon_counts_per_genstep.size() ; i++) x_total += photon_counts_per_genstep[i] ; 
    const NP* gs = QEvent::MakeCountGensteps(photon_counts_per_genstep) ; 

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs ); 
    gs->save(fold, "cngs.npy"); 

    return gs ; 
}

const NP* test_MakeCenterExtentGensteps(int nx, int ny, int nz)
{
    float4 ce = make_float4( 1.f, 2.f, 3.f, 100.f ); 
    uint4 cegs = make_uint4( nx, ny, nz, 100 ); 
    float gridscale = 1.f ; 

    const NP* gs = QEvent::MakeCenterExtentGensteps(ce, cegs, gridscale); 

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(BASE, create_dirs ); 
    gs->save(fold, "cegs.npy"); 

    return gs ; 
}

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

    //const NP* gs = test_MakeCountGensteps() ; 
    //const NP* gs = test_MakeCenterExtentGensteps(3, 0, 3) ; 
    const NP* gs = test_MakeCenterExtentGensteps(0, 0, 0) ; 
    assert( gs ); 
    gs->dump(); 

    //test_QEvent(gs); 
    return 0 ; 
}

