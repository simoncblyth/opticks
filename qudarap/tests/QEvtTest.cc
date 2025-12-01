/**
QEvtTest.cc
==============

~/o/qudarap/tests/QEvtTest.sh

TEST=one       ~/o/qudarap/tests/QEvtTest.sh
TEST=one VERBOSE=1  ~/o/qudarap/tests/QEvtTest.sh


TEST=sliced    ~/o/qudarap/tests/QEvtTest.sh
TEST=many      ~/o/qudarap/tests/QEvtTest.sh
TEST=loaded    ~/o/qudarap/tests/QEvtTest.sh
TEST=checkEvt  ~/o/qudarap/tests/QEvtTest.sh
TEST=quad6     ~/o/qudarap/tests/QEvtTest.sh

TEST=ALL       ~/o/qudarap/tests/QEvtTest.sh


**/


#include "OPTICKS_LOG.hh"

#include <csignal>
#include <cuda_runtime.h>

#include "OpticksGenstep.h"
#include "NP.hh"
#include "NPX.h"
#include "NPFold.h"

#include "QBuf.hh"
#include "QU.hh"

#include "ssys.h"
#include "spath.h"
#include "scuda.h"
#include "squad.h"
#include "sstamp.h"
#include "SGenstep.h"
#include "sphotonlite.h"

#include "SEvt.hh"
#include "SEvent.hh"
#include "QEvt.hh"

// some tests moved down to sysrap/tests/SEventTest.cc

struct QEvtTest
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




    static sevent* MockEventForMergeTest(const NP* p, const NP* l);
    static int PerLaunchMerge() ;


    template<typename T>
    static std::string GetPhotonSource();

    template<typename T>
    static int FinalMerge();

    template<typename T>
    static int FinalMerge_async();

    static int main();
};


const char* QEvtTest::TEST    = ssys::getenvvar("TEST", "many") ; // ALL: leads to cumulative OOM fail ?
bool        QEvtTest::VERBOSE = ssys::getenvbool("VERBOSE") ;

/**
QEvtTest::setGenstep_one
--------------------------------

1. QEvt::setGenstepUpload_NP one genstep to GPU
2. QEvt::gatherSeed downloads the resulting GPU seed buffer
3. compares downloaded seed buffer with expectations

**/

int QEvtTest::setGenstep_one()
{

    std::cout << "QEvtTest::setGenstep_one" << std::endl ;
    //                                             0  1  2  3  4  5  6  7  8
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    int x_num_photon = 0  ;
    NP* gs = SEvent::MakeCountGenstep(photon_counts_per_genstep, &x_num_photon ) ;

    LOG_IF(info, VERBOSE) << "[ new QEvt "  ;
    QEvt* event = new QEvt ;
    LOG_IF(info, VERBOSE) << "] new QEvt "  ;
    LOG_IF(info, VERBOSE) << " event.desc (bef QEvt::setGenstep) " << event->desc() ;


    LOG_IF(info, VERBOSE) << "[ QEvt::setGenstepUpload_NP "  ;
    event->setGenstepUpload_NP(gs);
    LOG_IF(info, VERBOSE) << "] QEvt::setGenstepUpload_NP "  ;

    LOG_IF(info, VERBOSE) << " event.desc (aft QEvt::setGenstep) " << event->desc() ;

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


int QEvtTest::setGenstep_sliced()
{
    std::cout << "QEvtTest::setGenstep_sliced" << std::endl ;

    //                                             0    1    2    3  4    5    6    7    8
    std::vector<int> photon_counts_per_genstep = { 300, 500, 200, 0, 100, 300, 400, 200, 400 };
    int tot_photon = 0  ;
    NP* gs = SEvent::MakeCountGenstep(photon_counts_per_genstep, &tot_photon ) ;

    QEvt* event = new QEvt ;
    LOG_IF(info, VERBOSE) << " event.desc (bef QEvt::setGenstep) " << event->desc() ;

    int max_slot = 1000 ;
    std::vector<sslice> gs_slice ;
    SGenstep::GetGenstepSlices( gs_slice, gs, max_slot );
    int num_slice = gs_slice.size();

    std::cout
        << "QEvtTest::setGenstep_sliced"
        << " tot_photon " << tot_photon
        << " max_slot " << max_slot
        << " num_slice " << num_slice
        << "\n"
        ;

    for(int i=0 ; i < num_slice ; i++)
    {
        const sslice& sl = gs_slice[i] ;

        event->setGenstepUpload_NP(gs, &sl );

        size_t num_photon = event->getNumPhoton();
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
        size_t num_seed = seed_->shape[0] ;

        assert( num_seed  == num_photon );
        int edgeitems = 100 ;
        LOG(info) << " seed: "   << SEvent::DescSeed(seed_v, num_seed, edgeitems );

#endif
    }
    return 0 ;
}










/**
QEvtTest::setGenstep_many
----------------------------

1. prepare num_v:3 gensteps together with corresponding photon and seed expectations
2. repeatedly upload those three gensteps followed by seed downloads and comparisons with expectations

**/

int QEvtTest::setGenstep_many()
{
    std::cout
        << "QEvtTest::setGenstep_many\n" ;

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

    QEvt* event = new QEvt ;

    int num_event = SEventConfig::NumEvent();
    int nj = num_v*num_event ;

    std::cout
        << "QEvtTest::setGenstep_many"
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
QEvtTest::setGenstep_loaded
-------------------------------

1. upload genstep with QEvt::setGenstepUpload_NP
2. check that get non-zero num_photon + num_simtrace


**/

int QEvtTest::setGenstep_loaded(NP* gs)
{
    std::cout
        << "[QEvtTest::setGenstep_loaded"
        << " gs " << ( gs ? gs->sstr() : "-" )
        << "\n"
        ;

    QEvt* event = new QEvt ;
    event->setGenstepUpload_NP(gs);

    unsigned num_photon = event->getNumPhoton() ;
    unsigned num_simtrace = event->getNumSimtrace() ;

    std::cout
        << "QEvtTest::setGenstep_loaded"
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
        << "]QEvtTest::setGenstep_loaded"
        << " gs " << ( gs ? gs->sstr() : "-" )
        << "\n"
        ;

    return 0 ;
}

int QEvtTest::setGenstep_loaded()
{
    std::cout << "QEvtTest::setGenstep_loaded" << std::endl ;
    const char* path_ = "$TMP/sysrap/SEventTest/cegs.npy" ;
    const char* path = spath::Resolve(path_);
    NP* gs0 = NP::Load(path);

    std::cout
        << "QEvtTest::setGenstep_loaded"
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


int QEvtTest::setGenstep_checkEvt()
{
    std::cout << "QEvtTest::setGenstep_checkEvt" << std::endl ;
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    int x_num_photon = SEvent::SumCounts( photon_counts_per_genstep ) ;
    std::cout << " x_num_photon " << x_num_photon << std::endl ;

    int x_total = 0 ;
    NP* gs = SEvent::MakeCountGenstep(photon_counts_per_genstep, &x_total) ;
    assert( x_num_photon == x_total ) ;

    QEvt* event = new QEvt ;
    event->setGenstepUpload_NP(gs);
    event->checkEvt();
    return 0 ;
}

int QEvtTest::setGenstep_quad6()
{
    std::cout << "QEvtTest::setGenstep_quad6" << std::endl ;

    std::vector<quad6> qgs(1) ;

    quad6& gs = qgs[0]  ;
    gs.q0.u = make_uint4( OpticksGenstep_CARRIER, 0u, 0u, 10u );
    gs.q1.u = make_uint4( 0u,0u,0u,0u );
    gs.q2.f = make_float4( 0.f, 0.f, 0.f, 0.f );    // post
    gs.q3.f = make_float4( 1.f, 0.f, 0.f, 1.f );    // dirw
    gs.q4.f = make_float4( 0.f, 1.f, 0.f, 500.f );  // polw
    gs.q5.f = make_float4( 0.f, 0.f, 0.f, 0.f );    // flag

    NP* a_gs = NPX::ArrayFromVec<float,quad6>( qgs, 6, 4) ;


    QEvt* event = new QEvt ;
    event->setGenstepUpload_NP( a_gs);

    event->gs->dump();
    event->checkEvt();

    cudaDeviceSynchronize();
    event->sev->save( "$TMP/QEvtTest/test_setGenstep_quad6" );

    return 0 ;
}


/**
QEvtTest::GetPhotonSource
--------------------------

To create source folder try::

    TEST=merge_M10 cxs_min.sh
    TEST=hitlitemerged ojt

**/



template<typename T>
std::string QEvtTest::GetPhotonSource()
{
    bool is_sphoton     = strcmp( T::NAME, "sphoton" ) == 0 ;
    bool is_sphotonlite = strcmp( T::NAME, "sphotonlite" ) == 0 ;
    assert( is_sphoton ^ is_sphotonlite );
    const char* src = is_sphotonlite ? "$AFOLD/photonlite.npy" : "$AFOLD/photon.npy" ;
    // these are placeholders for concat_hitlitemerged concat_hitmerged
    return src ;
}


sevent* QEvtTest::MockEventForMergeTest(const NP* p, const NP* l)
{
    sevent* evt = new sevent {} ;

    evt->photon = QU::UploadArray<sphoton>((sphoton*)p->bytes(), p->num_items(), "sphoton::MockupForMergeTest" ) ;
    evt->num_photon = p->num_items() ;

    evt->photonlite = QU::UploadArray<sphotonlite>((sphotonlite*)l->bytes(), l->num_items(), "sphotonlite::MockupForMergeTest" ) ;
    evt->num_photonlite = l->num_items() ;

    evt->hitmerged = nullptr ;
    evt->num_hitmerged = 0 ;

    evt->hitlitemerged = nullptr ;
    evt->num_hitlitemerged = 0 ;

    return evt ;
}


int QEvtTest::PerLaunchMerge()
{
    std::cout
        << "[QEvtTest::PerLaunchMerge\n"
        ;

    size_t ni = 1'000'000 ;

    NP* photon = sphoton::MockupForMergeTest(ni);
    NP* photonlite = sphotonlite::MockupForMergeTest(ni);

    sevent* evt = MockEventForMergeTest(photon, photonlite);

    cudaStream_t stream = 0 ;
    NP* hitmerged = QEvt::PerLaunchMerge<sphoton>(evt, stream );
    NP* hitlitemerged = QEvt::PerLaunchMerge<sphotonlite>(evt, stream );

    std::cout
        << " evt.num_photon     " << evt->num_photon << "\n"
        << " evt.num_photonlite " << evt->num_photonlite << "\n"
        << " evt.num_hitmerged  " << evt->num_hitmerged << "\n"
        << " evt.num_hitlitemerged  " << evt->num_hitlitemerged << "\n"
        << " evt.hitmerged " << ( evt->hitmerged ? "YES" : "NO " ) << "\n"
        << " evt.hitlitemerged " << ( evt->hitlitemerged ? "YES" : "NO " ) << "\n"
        << " hitmerged " << ( hitmerged ? hitmerged->sstr() : "-" ) << "\n"
        << " hitlitemerged " << ( hitlitemerged ? hitlitemerged->sstr() : "-" ) << "\n"
        ;


    NPFold* f = new NPFold ;
    f->add("photon", photon);
    f->add("photonlite", photonlite);
    f->add("hitmerged", hitmerged );
    f->add("hitlitemerged", hitlitemerged );
    f->save("$FOLD"); // includes TEST in last elem

    std::cout
       << "]QEvtTest::PerLaunchMerge"
       << "\n"
       ;

    return 0 ;
}




template<typename T>
int QEvtTest::FinalMerge()
{
    std::string src_ = GetPhotonSource<T>();
    const char* src = src_.c_str();

    const NP* all = NP::Load(src) ;
    std::cout << "[QEvtTest::FinalMerge all " << ( all ? all->sstr() : "-" ) << "\n" ;
    std::cout << " all\n" << T::Desc(all, 10) ;

    cudaStream_t producer ;
    cudaStreamCreate(&producer);

    NP* hit = QEvt::FinalMerge<T>(all, producer );
    std::cout
        << "QEvtTest::FinalMerge_async"
        << " all " << ( all ? all->sstr() : "-" )
        << " hit " << ( hit ? hit->sstr() : "-" )
        << "\n"
        ;

    std::cout << " hit\n" << T::Desc(hit, 10) ;
    return 0 ;
}


template<typename T>
int QEvtTest::FinalMerge_async()
{
    std::string src_ = GetPhotonSource<T>();
    const char* src = src_.c_str();

    const NP* all = NP::Load(src) ;
    std::cout << "[QEvtTest::FinalMerge_async all " << ( all ? all->sstr() : "-" ) << "\n" ;

    std::cout << " all\n" << T::Desc(all, 10) ;

    cudaStream_t producer ;
    cudaStreamCreate(&producer);

    NP_future producer_result = QEvt::FinalMerge_async<T>(all, producer );

    cudaStream_t consumer ;
    cudaStreamCreate(&consumer);

    // make consumer wait for producer_result.ready event
    producer_result.wait(consumer);

    NP* hit = producer_result.arr ;

    std::cout
        << "QEvtTest::FinalMerge_async"
        << " T::NAME " << T::NAME
        << " src " << ( src ? src : "-" )
        << " all " << ( all ? all->sstr() : "-" )
        << " hit " << ( hit ? hit->sstr() : "-" )
        << "\n"
        ;

    std::cout << " hit\n" << T::Desc(hit, 10) ;

    return 0 ;
}


int QEvtTest::main()
{
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    std::cout << "[QEvtTest::main ALL " << ( ALL ? "YES" : "NO " ) << " TEST [" << ( TEST ? TEST : "-" ) << "]\n" ;

    if(ALL||0==strcmp(TEST,"one"))      rc += setGenstep_one();
    if(ALL||0==strcmp(TEST,"sliced"))   rc += setGenstep_sliced();
    if(ALL||0==strcmp(TEST,"many"))     rc += setGenstep_many();
    if(ALL||0==strcmp(TEST,"loaded"))   rc += setGenstep_loaded();
    if(ALL||0==strcmp(TEST,"checkEvt")) rc += setGenstep_checkEvt();
    if(ALL||0==strcmp(TEST,"quad6"))    rc += setGenstep_quad6();

    if(ALL||0==strcmp(TEST,"PerLaunchMerge")) rc += PerLaunchMerge();

    if(ALL||0==strcmp(TEST,"LiteFinalMerge"))          rc += FinalMerge<sphotonlite>();
    if(ALL||0==strcmp(TEST,"LiteFinalMerge_async"))    rc += FinalMerge_async<sphotonlite>();
    if(ALL||0==strcmp(TEST,"FullFinalMerge"))          rc += FinalMerge<sphoton>();
    if(ALL||0==strcmp(TEST,"FullFinalMerge_async"))    rc += FinalMerge_async<sphoton>();


    std::cout << "]QEvtTest::main rc [" << rc << "]\n" ;
    return rc  ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    SEvt::Create(SEvt::EGPU) ;

    return QEvtTest::main();
}

