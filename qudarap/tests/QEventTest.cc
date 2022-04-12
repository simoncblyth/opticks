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
    //                                             0  1  2  3  4  5  6  7  8  
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    unsigned x_num_photon = SEvent::SumCounts( photon_counts_per_genstep ) ; 
    const NP* gs = SEvent::MakeCountGensteps(photon_counts_per_genstep) ; 

    std::vector<int> x_seed ; 
    SEvent::ExpectedSeeds(x_seed, photon_counts_per_genstep); 
    assert( x_seed.size() == x_num_photon ); 


    QEvent* event = new QEvent ; 
    //LOG(info) << " event.desc (bef QEvent::setGensteps) " << event->desc() ; 
    event->setGensteps(gs); 
    //LOG(info) << " event.desc (aft QEvent::setGensteps) " << event->desc() ; 

    unsigned num_photon = event->getNumPhoton();  
    //LOG(info) << " num_photon " << num_photon << " x_num_photon " << x_num_photon ; 
    assert( x_num_photon == num_photon ); 

    std::vector<int> seed ; 
    event->downloadSeed(seed); 
    assert( seed.size() == num_photon ); 

    int seed_mismatch = SEvent::CompareSeeds( seed, x_seed ); 

    LOG(info) << " seed: "   << QEvent::DescSeed(seed, 100 ); 
    LOG(info) << " x_seed: " << QEvent::DescSeed(x_seed, 100 ); 
    LOG(info) << " seed_mismatch " << seed_mismatch ; 
}

void test_setGensteps_1()
{
    const int num_v = 3 ; 

    typedef std::vector<int> VI ; 
    VI* v = new VI[num_v] ; 
    VI* x_seed = new VI[num_v] ; 

    NP** gs = new NP*[num_v] ; 
    int* x_num_photon = new int[num_v] ; 

    v[0] = {3, 5, 2, 0, 1, 3, 4, 2, 4 } ; 
    v[1] = { 30, 50, 20, 0, 10, 30, 40, 20, 40 };
    v[2] = { 300, 500, 200, 0, 100, 300, 400, 200, 400 } ; 

    for(int i=0 ; i < num_v ; i++)
    {
        x_num_photon[i] = SEvent::SumCounts( v[i] ) ; 
        gs[i] = SEvent::MakeCountGensteps( v[i] ); 

        SEvent::ExpectedSeeds(x_seed[i], v[i]); 
        assert( int(x_seed[i].size()) == x_num_photon[i] ); 

        std::cout 
            << " i " << std::setw(3) << i 
            << " x_num_photon[i] " << std::setw(5) << x_num_photon[i] 
            << " gs[i] " << gs[i]->desc() 
            << std::endl 
            << " x_seed: " << QEvent::DescSeed(x_seed[i], 100 )
            << std::endl 
            ; 
    }

    QEvent* event = new QEvent ; 

    int nj = num_v*10 ; 
    //int nj = 2 ; 

    for(int j=0 ; j < nj ; j++)
    {
        int i = j % 3 ; 

        event->setGensteps(gs[i]); 

        int num_photon = event->getNumPhoton();  
        assert( x_num_photon[i] == num_photon ); 

        std::vector<int> seed ; 
        event->downloadSeed(seed); 
        assert( int(seed.size()) == num_photon ); 
        int seed_mismatch = SEvent::CompareSeeds( seed, x_seed[i] ); 
 
        std::cout 
            << " j " << std::setw(3) << j 
            << " i " << std::setw(3) << i 
            << " x_num_photon[i] " << std::setw(5) << x_num_photon[i] 
            << " num_photon " << std::setw(5) << num_photon 
            << " seed_mismatch " << std::setw(5) << seed_mismatch 
            << event->desc() 
            << std::endl 
            ;

        if( seed_mismatch > 0 )
        { 
            std::cout 
                << " seed: " << QEvent::DescSeed(seed, 100 )
                << std::endl 
                ; 
        }
        assert( seed_mismatch == 0 ); 

    }
}

void test_setGensteps_2(const NP* gs)
{
    QEvent* event = new QEvent ; 
    event->setGensteps(gs); 

    unsigned num_photon = event->getNumPhoton() ; 
    assert( num_photon > 0); 

    LOG(info) << event->desc() ; 
    //event->seed->download_dump("event->seed", 10); 
    event->checkEvt(); 
}

void test_setGensteps_2()
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
    test_setGensteps_2(gs0); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //test_setGensteps_0(); 
    test_setGensteps_1(); 

    cudaDeviceSynchronize(); 
    return 0 ; 
}

