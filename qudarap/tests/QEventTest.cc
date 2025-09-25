/**
QEventTest.cc
==============

~/o/qudarap/tests/QEventTest.sh

TEST=one       ~/o/qudarap/tests/QEventTest.sh
TEST=one VERBOSE=1  ~/o/qudarap/tests/QEventTest.sh


TEST=sliced    ~/o/qudarap/tests/QEventTest.sh
TEST=many      ~/o/qudarap/tests/QEventTest.sh
TEST=loaded    ~/o/qudarap/tests/QEventTest.sh
TEST=checkEvt  ~/o/qudarap/tests/QEventTest.sh
TEST=quad6     ~/o/qudarap/tests/QEventTest.sh

TEST=ALL       ~/o/qudarap/tests/QEventTest.sh


**/


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
#include "SGenstep.h"

#include "SEvt.hh"
#include "SEvent.hh"
#include "QEvent.hh"

// some tests moved down to sysrap/tests/SEventTest.cc

struct QEventTest
{
    static constexpr const int M = 1000000 ;
    static const char* TEST ;
    static bool VERBOSE ;

    static int setGenstep_one() ;
    static int setGenstep_sliced() ;
    static int setGenstep_many() ;
    static int setGenstep_loaded(NP* gs);
    static int setGenstep_loaded();
    static int setGenstep_checkEvt();
    static int setGenstep_quad6();

    static int main();
};


const char* QEventTest::TEST    = ssys::getenvvar("TEST", "many") ; // ALL: leads to cumulative OOM fail ?
bool        QEventTest::VERBOSE = ssys::getenvbool("VERBOSE") ;

/**
QEventTest::setGenstep_one
--------------------------------

1. QEvent::setGenstepUpload_NP one genstep to GPU
2. QEvent::gatherSeed downloads the resulting GPU seed buffer
3. compares downloaded seed buffer with expectations

**/

int QEventTest::setGenstep_one()
{

    std::cout << "QEventTest::setGenstep_one" << std::endl ;
    //                                             0  1  2  3  4  5  6  7  8
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    int x_num_photon = 0  ;
    NP* gs = SEvent::MakeCountGenstep(photon_counts_per_genstep, &x_num_photon ) ;

    LOG_IF(info, VERBOSE) << "[ new QEvent "  ;
    QEvent* event = new QEvent ;
    LOG_IF(info, VERBOSE) << "] new QEvent "  ;
    LOG_IF(info, VERBOSE) << " event.desc (bef QEvent::setGenstep) " << event->desc() ;


    LOG_IF(info, VERBOSE) << "[ QEvent::setGenstepUpload_NP "  ;
    event->setGenstepUpload_NP(gs);
    LOG_IF(info, VERBOSE) << "] QEvent::setGenstepUpload_NP "  ;

    LOG_IF(info, VERBOSE) << " event.desc (aft QEvent::setGenstep) " << event->desc() ;

    int num_photon = event->getNumPhoton();
    bool num_photon_expect = x_num_photon == num_photon ;
    LOG(info) << " num_photon " << num_photon << " x_num_photon " << x_num_photon ;
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
    return 0 ;
}


int QEventTest::setGenstep_sliced()
{
    std::cout << "QEventTest::setGenstep_sliced" << std::endl ;

    //                                             0    1    2    3  4    5    6    7    8
    std::vector<int> photon_counts_per_genstep = { 300, 500, 200, 0, 100, 300, 400, 200, 400 };
    int tot_photon = 0  ;
    NP* gs = SEvent::MakeCountGenstep(photon_counts_per_genstep, &tot_photon ) ;

    QEvent* event = new QEvent ;
    LOG_IF(info, VERBOSE) << " event.desc (bef QEvent::setGenstep) " << event->desc() ;

    int max_slot = 1000 ;
    std::vector<sslice> gs_slice ;
    SGenstep::GetGenstepSlices( gs_slice, gs, max_slot );
    int num_slice = gs_slice.size();

    std::cout
        << "QEventTest::setGenstep_sliced"
        << " tot_photon " << tot_photon
        << " max_slot " << max_slot
        << " num_slice " << num_slice
        << "\n"
        ;

    for(int i=0 ; i < num_slice ; i++)
    {
        const sslice& sl = gs_slice[i] ;

        event->setGenstepUpload_NP(gs, &sl );

        int num_photon = event->getNumPhoton();
        bool num_photon_expect = sl.ph_count == num_photon ;

        std::cout
            << " i " << std::setw(3) << i
            << " sl " << sl.desc()
            << " num_photon " << num_photon
            << "\n"
            ;

        assert( num_photon_expect );
        if(!num_photon_expect) std::raise(SIGINT);

#ifndef PRODUCTION

        NP* seed_ = event->gatherSeed();
        const int* seed_v = seed_->values<int>();
        int num_seed = seed_->shape[0] ;

        assert( num_seed  == num_photon );
        int edgeitems = 100 ;
        LOG(info) << " seed: "   << SEvent::DescSeed(seed_v, num_seed, edgeitems );

#endif
    }
    return 0 ;
}










/**
QEventTest::setGenstep_many
----------------------------

1. prepare num_v:3 gensteps together with corresponding photon and seed expectations
2. repeatedly upload those three gensteps followed by seed downloads and comparisons with expectations

**/

int QEventTest::setGenstep_many()
{
    std::cout
        << "QEventTest::setGenstep_many\n" ;

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
    int nj = num_v*num_event ;

    std::cout
        << "QEventTest::setGenstep_many"
        << " num_event " << num_event
        << " num_v " << num_v
        << " nj " << nj
        << "\n"
        ;

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
    return 0 ;
}

/**
QEventTest::setGenstep_loaded
-------------------------------

1. upload genstep with QEvent::setGenstepUpload_NP
2. check that get non-zero num_photon + num_simtrace


**/

int QEventTest::setGenstep_loaded(NP* gs)
{
    std::cout
        << "[QEventTest::setGenstep_loaded"
        << " gs " << ( gs ? gs->sstr() : "-" )
        << "\n"
        ;

    QEvent* event = new QEvent ;
    event->setGenstepUpload_NP(gs);

    unsigned num_photon = event->getNumPhoton() ;
    unsigned num_simtrace = event->getNumSimtrace() ;

    std::cout
        << "QEventTest::setGenstep_loaded"
        << " num_photon " << num_photon
        << " num_photon/M " << num_photon/M
        << " num_simtrace " << num_simtrace
        << " num_simtrace/M " << num_simtrace/M
        << "\n"
        ;

    bool num_photon_expect = num_photon + num_simtrace > 0 ;
    assert( num_photon_expect );
    if(!num_photon_expect) std::raise(SIGINT);

    LOG(info) << event->desc() ;
    //event->seed->download_dump("event->seed", 10);
    event->checkEvt();

    std::cout
        << "]QEventTest::setGenstep_loaded"
        << " gs " << ( gs ? gs->sstr() : "-" )
        << "\n"
        ;

    return 0 ;
}

int QEventTest::setGenstep_loaded()
{
    std::cout << "QEventTest::setGenstep_loaded" << std::endl ;
    const char* path_ = "$TMP/sysrap/SEventTest/cegs.npy" ;
    const char* path = spath::Resolve(path_);
    NP* gs0 = NP::Load(path);

    std::cout
        << "QEventTest::setGenstep_loaded"
        << " path " << ( path ? path : "-" )
        << " gs0 " << ( gs0 ? gs0->sstr() : "-" )
        << "\n"
        ;


    if( gs0 == nullptr )
    {
        LOG(fatal)
            << "failed to load from"
            << " path_ " << path_
            << " path " << path
            ;
        //assert(0);
        return 0 ;
    }

    gs0->dump();
    setGenstep_loaded(gs0);
    return 0 ;
}


int QEventTest::setGenstep_checkEvt()
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
    return 0 ;
}

int QEventTest::setGenstep_quad6()
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
    event->setGenstepUpload_NP( a_gs);

    event->gs->dump();
    event->checkEvt();

    cudaDeviceSynchronize();
    event->sev->save( "$TMP/QEventTest/test_setGenstep_quad6" );

    return 0 ;
}


int QEventTest::main()
{
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    std::cout << "[QEventTest::main ALL " << ( ALL ? "YES" : "NO " ) << "\n" ;

    if(ALL||0==strcmp(TEST,"one"))      rc += setGenstep_one();
    if(ALL||0==strcmp(TEST,"sliced"))   rc += setGenstep_sliced();
    if(ALL||0==strcmp(TEST,"many"))     rc += setGenstep_many();
    if(ALL||0==strcmp(TEST,"loaded"))   rc += setGenstep_loaded();
    if(ALL||0==strcmp(TEST,"checkEvt")) rc += setGenstep_checkEvt();
    if(ALL||0==strcmp(TEST,"quad6"))    rc += setGenstep_quad6();

    std::cout << "]QEventTest::main rc [" << rc << "]\n" ;
    return rc  ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    SEvt::Create(SEvt::EGPU) ;

    return QEventTest::main();
}

