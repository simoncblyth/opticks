#include "OPTICKS_LOG.hh"

#include <csignal>
#include <cuda_runtime.h>

#include "OpticksGenstep.h"
#include "NP.hh"
#include "NPX.h"

#include "QBuf.hh"

#include "ssys.h"
#include "spath.h"
#include "scuda.h"
#include "squad.h"
#include "sstamp.h"

#include "SEvt.hh"
#include "SEvent.hh"
#include "QEvent.hh"

// some tests moved down to sysrap/tests/SEventTest.cc

struct QEventTest
{ 
    static void setGenstep_one() ; 
    static void setGenstep_many() ; 
    static void setGenstep_loaded(NP* gs); 
    static void setGenstep_loaded(); 
    static void setGenstep_checkEvt(); 
    static void setGenstep_quad6(); 

    static int main(); 
};

/**
QEventTest::setGenstep_one
--------------------------------

**/

void QEventTest::setGenstep_one()
{
    std::cout << "QEventTest::setGenstep_one" << std::endl ; 
    //                                             0  1  2  3  4  5  6  7  8  
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    int x_num_photon = 0  ; 
    NP* gs = SEvent::MakeCountGenstep(photon_counts_per_genstep, &x_num_photon ) ; 


    QEvent* event = new QEvent ; 
    //LOG(info) << " event.desc (bef QEvent::setGenstep) " << event->desc() ; 
    event->setGenstepUpload_NP(gs); 
    //LOG(info) << " event.desc (aft QEvent::setGenstep) " << event->desc() ; 

    int num_photon = event->getNumPhoton();  
    bool num_photon_expect = x_num_photon == num_photon ;
    //LOG(info) << " num_photon " << num_photon << " x_num_photon " << x_num_photon ; 
    assert( num_photon_expect ); 
    if(!num_photon_expect) std::raise(SIGINT); 

#ifndef PRODUCTION

    std::vector<int> xseed ; 
    SEvent::ExpectedSeeds(xseed, photon_counts_per_genstep); 
    assert( int(xseed.size()) == x_num_photon ); 
    const int* xseed_v = xseed.data(); 


    NP* seed = event->gatherSeed(); 
    const int* seed_v = seed->values<int>(); 
    int num_seed = seed->shape[0] ; 

    assert( num_seed == num_photon ); 

    int seed_mismatch = SEvent::CompareSeeds( seed_v, xseed_v, num_seed ); 

    LOG(info) << " seed: "   << SEvent::DescSeed(seed_v, num_seed, 100 ); 
    LOG(info) << " x_seed: " << SEvent::DescSeed(xseed_v, num_seed, 100 ); 
    LOG(info) << " seed_mismatch " << seed_mismatch ; 
#endif
}

void QEventTest::setGenstep_many()
{
    std::cout << "QEventTest::setGenstep_many" << std::endl ; 
    const int num_v = 3 ; 

    typedef std::vector<int> VI ; 
    VI* v = new VI[num_v] ; 
    VI* x_seed = new VI[num_v] ; 

    NP** gs = new NP*[num_v] ; 
    int* x_num_photon = new int[num_v] ; 
    int* x_total = new int[num_v] ; 

    v[0] = {3, 5, 2, 0, 1, 3, 4, 2, 4 } ; 
    v[1] = { 30, 50, 20, 0, 10, 30, 40, 20, 40 };
    v[2] = { 300, 500, 200, 0, 100, 300, 400, 200, 400 } ; 

    for(int i=0 ; i < num_v ; i++)
    {
        x_num_photon[i] = SEvent::SumCounts( v[i] ) ; 
        gs[i] = SEvent::MakeCountGenstep( v[i], &x_total[i] ); 
        assert( x_total[i] == x_num_photon[i] ); 

        SEvent::ExpectedSeeds(x_seed[i], v[i]); 
        assert( int(x_seed[i].size()) == x_num_photon[i] ); 

        std::cout 
            << " i " << std::setw(3) << i 
            << " x_num_photon[i] " << std::setw(5) << x_num_photon[i] 
            << " gs[i] " << gs[i]->desc() 
            << std::endl 
            << " x_seed: " << SEvent::DescSeed(x_seed[i].data(), x_seed[i].size(), 100 )
            << std::endl 
            ; 
    }

    QEvent* event = new QEvent ; 

    int num_event = SEventConfig::NumEvent(); 

    std::cout << " num_event " << num_event << std::endl ;

    int nj = num_v*num_event ; 
    //int nj = 2 ; 

    for(int j=0 ; j < nj ; j++)
    {
        int i = j % 3 ; 

        event->setGenstepUpload_NP(gs[i]); 

        int num_photon = event->getNumPhoton();  
        bool num_photon_expect = x_num_photon[i] == num_photon ;
        assert( num_photon_expect ); 
        if(!num_photon_expect) std::raise(SIGINT); 


#ifndef PRODUCTION
        sstamp::sleep_us(100000); 

        NP* seed_ = event->gatherSeed(); 
        const int* seed = seed_->values<int>();   
        int num_seed = seed_->shape[0] ;

        assert( num_seed  == num_photon ); 
        int seed_mismatch = SEvent::CompareSeeds( seed, x_seed[i].data(), x_seed[i].size() ); 
 
        std::cout 
            << " j " << std::setw(3) << j 
            << " i " << std::setw(3) << i 
            << " x_num_photon[i] " << std::setw(5) << x_num_photon[i] 
            << " num_photon " << std::setw(5) << num_photon 
            << " seed_mismatch " << std::setw(5) << seed_mismatch 
           //  << event->desc() 
            << std::endl 
            ;


        if( seed_mismatch > 0 )
        { 
            std::cout 
                << " seed: " << SEvent::DescSeed(seed, num_seed, 100 )
                << std::endl 
                ; 
        }
        assert( seed_mismatch == 0 ); 
#endif

    }
}

void QEventTest::setGenstep_loaded(NP* gs)
{
    QEvent* event = new QEvent ; 
    event->setGenstepUpload_NP(gs); 

    unsigned num_photon = event->getNumPhoton() ; 
    bool num_photon_expect = num_photon > 0 ;
    assert( num_photon_expect ); 
    if(!num_photon_expect) std::raise(SIGINT); 

    LOG(info) << event->desc() ; 
    //event->seed->download_dump("event->seed", 10); 
    event->checkEvt(); 
}

void QEventTest::setGenstep_loaded()
{
    std::cout << "QEventTest::setGenstep_loaded" << std::endl ; 
    const char* path_ = "$TMP/sysrap/SEventTest/cegs.npy" ;
    const char* path = spath::Resolve(path_); 
    NP* gs0 = NP::Load(path); 

    if( gs0 == nullptr )
    {
        LOG(fatal) 
            << "failed to load from"
            << " path_ " << path_
            << " path " << path 
            ;
        //assert(0); 
        return ; 
    }

    gs0->dump(); 
    setGenstep_loaded(gs0); 
}


void QEventTest::setGenstep_checkEvt()
{
    std::cout << "QEventTest::setGenstep_checkEvt" << std::endl ; 
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    int x_num_photon = SEvent::SumCounts( photon_counts_per_genstep ) ; 
    std::cout << " x_num_photon " << x_num_photon << std::endl ; 

    int x_total = 0 ; 
    NP* gs = SEvent::MakeCountGenstep(photon_counts_per_genstep, &x_total) ; 
    assert( x_num_photon == x_total ) ; 

    QEvent* event = new QEvent ; 
    event->setGenstepUpload_NP(gs); 
    event->checkEvt(); 
}

void QEventTest::setGenstep_quad6()
{
    std::cout << "QEventTest::setGenstep_quad6" << std::endl ; 

    std::vector<quad6> qgs(1) ;

    quad6& gs = qgs[0]  ; 
    gs.q0.u = make_uint4( OpticksGenstep_CARRIER, 0u, 0u, 10u );   
    gs.q1.u = make_uint4( 0u,0u,0u,0u );  
    gs.q2.f = make_float4( 0.f, 0.f, 0.f, 0.f );    // post
    gs.q3.f = make_float4( 1.f, 0.f, 0.f, 1.f );    // dirw
    gs.q4.f = make_float4( 0.f, 1.f, 0.f, 500.f );  // polw
    gs.q5.f = make_float4( 0.f, 0.f, 0.f, 0.f );    // flag

    NP* a_gs = NPX::ArrayFromVec<float,quad6>( qgs, 6, 4) ;


    QEvent* event = new QEvent ; 
    event->setGenstepUpload_NP( a_gs );  

    event->gs->dump(); 
    event->checkEvt(); 

    cudaDeviceSynchronize(); 
    event->sev->save( "$TMP/QEventTest/test_setGenstep_quad6" ); 
}



int QEventTest::main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL") ; 
    if(strcmp(TEST, "ALL") == 0 )
    {
        setGenstep_one(); 
        setGenstep_many(); 
        setGenstep_loaded(); 
        setGenstep_checkEvt(); 
        setGenstep_quad6(); 
        setGenstep_many(); 
    }
    else if( strcmp(TEST, "setGenstep_many") == 0 )
    {
        setGenstep_many(); 
    }
    return 0 ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt::Create(SEvt::EGPU) ;

    return QEventTest::main(); 
}

