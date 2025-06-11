// ~/opticks/sysrap/tests/SEvtTest.sh
#include <csignal>

#include "OPTICKS_LOG.hh"
#include "OpticksGenstep.h"

#include "scuda.h"
#include "spath.h"
#include "sprof.h"
#include "stran.h"
#include "sdirectory.h"
#include "ssys.h"

#include "SEvent.hh"
#include "SEventConfig.hh"
#include "SEvt.hh"
#include "NPFold.h"


struct SEvtTest
{
    static constexpr const int M = 1000000 ;
    static const char* TEST ;

    static int AddGenstep();
    static int LifeCycle();
    static int InputPhoton();
    static int getSaveDir();
    static int getDir();
    static int setMetaProf();
    static int hostside_running_resize_();
    static int CountNibbles();
    static int makeGenstepArrayFromVector();
    static int saveExtra();

    static int Main(int argc, char** argv);

};


const char* SEvtTest::TEST = ssys::getenvvar("TEST", "CountNibbles" );

int SEvtTest::AddGenstep()
{
    SEvt* evt = SEvt::Create(0) ;
    bool evt_expect = SEvt::Get(0) == evt ;
    assert( evt_expect );
    if(!evt_expect) std::raise(SIGINT);

    for(unsigned i=0 ; i < 10 ; i++)
    {
        quad6 q ;
        q.set_numphoton(1000) ;
        unsigned gentype = i % 2 == 0 ? OpticksGenstep_SCINTILLATION : OpticksGenstep_CERENKOV ;
        q.set_gentype(gentype);

        SEvt::AddGenstep(q);
    }

    std::cout << SEvt::Get(0)->desc() << std::endl ;
    return 0 ;
}


int SEvtTest::LifeCycle()
{
    unsigned max_bounce = 9 ;
    SEventConfig::SetMaxBounce(max_bounce);
    SEventConfig::SetMaxRecord(max_bounce+1);
    SEventConfig::SetMaxRec(max_bounce+1);
    SEventConfig::SetMaxSeq(1);

    SEvt* evt = SEvt::Create(0) ;

    evt->setIndex(214);
    evt->unsetIndex();

    quad6 gs ;
    gs.set_numphoton(1) ;
    gs.set_gentype(OpticksGenstep_TORCH);

    evt->addGenstep(gs);

    spho label = {0,0,0,{0,0,0,0}} ;

    evt->beginPhoton(label);

    int bounce0 = 0 ;
    int bounce1 = 0 ;

    bounce0 = evt->slot[label.id] ;
    evt->pointPhoton(label);
    bounce1 = evt->slot[label.id] ;

    bool bounce_expect = bounce1 == bounce0 + 1 ;
    assert( bounce_expect );
    if(!bounce_expect) std::raise(SIGINT);

    std::cout
         << " i " << std::setw(3) << -1
         << " bounce0 " << std::setw(3) << bounce0
         << " : " << evt->current_ctx.p.descFlag()
         << std::endl
         ;


    std::vector<unsigned> history = {
       BOUNDARY_TRANSMIT,
       BOUNDARY_TRANSMIT,
       BULK_SCATTER,
       BULK_SCATTER,
       BULK_SCATTER,
       BULK_SCATTER,
       BULK_SCATTER,
       BULK_SCATTER,
       BULK_SCATTER,
       BULK_SCATTER,
       BULK_SCATTER,
       BULK_SCATTER,
       BULK_REEMIT,
       BOUNDARY_TRANSMIT,
       SURFACE_DETECT
    } ;

    for(int i=0 ; i < int(history.size()) ; i++)
    {
        unsigned flag = history[i] ;
        evt->current_ctx.p.set_flag(flag);

        bounce0 = evt->slot[label.id] ;
        evt->pointPhoton(label);
        bounce1 = evt->slot[label.id] ;
        assert( bounce1 == bounce0 + 1 );

        std::cout
             << " i " << std::setw(3) << i
             << " bounce0 " << std::setw(3) << bounce0
             << " : " << evt->current_ctx.p.descFlag()
             << std::endl
             ;
    }

    evt->finalPhoton(label);

    evt->save("$TMP/SEvtTest");
    LOG(info) << evt->desc() ;
    return 0 ;
}

int SEvtTest::InputPhoton()
{
    const char* ipf = SEventConfig::InputPhotonFrame();
    if( ipf == nullptr) return 1 ;

    SEvt* evt = SEvt::Create(0) ;
    LOG(info) << evt->desc() ;

    NP* ip = evt->getInputPhoton();
    LOG_IF(fatal, !ip ) << " FAILED TO getInputPhoton " ;
    if(!ip) return 1 ;


    const char* FOLD = spath::Resolve("$TMP/SEvtTest/test_InputPhoton");
    sdirectory::MakeDirs( FOLD, 0 );

    ip->save(FOLD, spath::Name("ipf", ipf, ".npy") );

    /*
    const qat4* q = SEvt::CF->getInputPhotonFrame();
    Tran<double>* tr = Tran<double>::ConvertToTran(q);

    NP* fr = NP::Make<float>(1,4,4);
    memcpy( fr->bytes(), q->cdata(), fr->arr_bytes() );
    fr->save(FOLD, SStr::Name("fr", ipf, ".npy" ));

    tr->save( FOLD, SStr::Name("tr",ipf, ".npy" )) ;

    */
    return 0 ;
}


/**
test_savedir
===============

savedir1 examples::

    /tmp/blyth/opticks/GEOM/SEvtTest/ALL
    /tmp/blyth/opticks/GEOM/Cheese/SEvtTest/ALL    # when GEOM envvar is Cheese

Only after the save does the savedir get set.

**/


int SEvtTest::getSaveDir()
{
    SEvt* evt = SEvt::Create(0);

    //LOG(info) << evt->desc() ;

    const char* savedir0 = evt->getSaveDir() ;
    assert( savedir0 == nullptr );

    evt->save();

    const char* savedir1 = evt->getSaveDir() ;
    //assert( savedir1 != nullptr );

    LOG(info)
        << " savedir0 " << ( savedir0 ? savedir0 : "(null)" )
        << " savedir1 " << ( savedir1 ? savedir1 : "(null)" )
        ;
    return 0 ;
}


int SEvtTest::getDir()
{
    SEvt* evt = SEvt::Create(0);
    const char* dir0 = evt->getDir();

    LOG(info)
        << "getDir" << std::endl
        << " dir0 [" << ( dir0 ? dir0 : "-" ) << "]" << std::endl
        ;
    return 0 ;
}


int SEvtTest::setMetaProf()
{
    SEvt* evt = SEvt::Create(0);

    sprof prof = {} ;
    sprof::Stamp(prof);
    evt->setMetaProf("test_setMeta", prof );

    std::cout << "evt->meta" << std::endl << evt->meta << std::endl ;
    return 0 ;
}

/**
SEvtTest::hostside_running_resize_
------------------------------------

Profile time, VM, RSS change from hostside_running_resize

**/

int SEvtTest::hostside_running_resize_()
{
    int num = 10*M ;
    bool edump = false ;

    SEventConfig::SetMaxPhoton(num);

    SEvt::Create_EGPU() ;
    SEvt* evt = SEvt::Get_EGPU();

    sprof p0, p1, p2  ;

    sprof::Stamp(p0);

    evt->setNumPhoton(num);

    if(edump)
    std::cout
        << " (SEvt)evt->(sevent)evt->descNum "
        << evt->evt->descNum()
        ;

    evt->hostside_running_resize_() ;

    sprof::Stamp(p1);
    std::cout
        << "sprof::Desc(p0, p1) : before and after setNumPhoton+hostside_running_resize_ : to " << num << std::endl
        << sprof::Desc(p0, p1 )
        ;

    evt->clear_output() ;

    sprof::Stamp(p2);
    std::cout
        << "sprof::Desc(p1,p2) : before and after : SEvt::clear_vectors  " << std::endl
        << "SMALL DELTA INDICATES THE RESIZE TO ZERO : DID NOT DEALLOCATE MEMORY " << std::endl
        << "FIND THAT NEED shrink = true TO GET THE DEALLOC TO HAPPEN " << std::endl
        << sprof::Desc(p1, p2 )
        ;

    return 0 ;
}

int SEvtTest::CountNibbles()
{
    const char* path = spath::Resolve("$SEQPATH");
    NP* seq = path ? NP::LoadIfExists(path) : nullptr ;
    if(seq == nullptr) return 0 ;

    NP* seqnib = SEvt::CountNibbles(seq);
    NP* seqnib_table = SEvt::CountNibbles_Table(seqnib);

    std::cout
        << "SEvtTest::CountNibbles"
        << std::endl
        << " path " << ( path ? path : "-" )
        << std::endl
        << " seq " << ( seq ? seq->sstr() : "-" )
        << std::endl
        << " seqnib " << ( seqnib ? seqnib->sstr() : "-" )
        << std::endl
        << " seqnib_table " << ( seqnib_table ? seqnib_table->sstr() : "-" )
        << std::endl
        ;

    std::cout << seqnib_table->descTable<int>(7) ;

    NPFold* fold = new NPFold ;
    fold->add( "seqnib", seqnib );
    fold->add( "seqnib_table", seqnib_table );
    fold->save("$FOLD");
    return 0 ;
}

int SEvtTest::makeGenstepArrayFromVector()
{
    std::cout << "[ SEvtTest::makeGenstepArrayFromVector \n";

    SEvt::Create_EGPU() ;
    SEvt* evt = SEvt::Get_EGPU();

    NP* gs0 = evt->makeGenstepArrayFromVector(); // get array from the genstep vector
    assert( gs0 == nullptr && "before adding genstep to vector expect null ") ;

    const NP* igs = SEvent::MakeDemoGenstep("torch");
    evt->addGenstep(igs);  // converts into quad6 which are added to vector

    NP* gs = evt->makeGenstepArrayFromVector(); // get array from the genstep vector
    assert( gs != nullptr && "after adding genstep to vector expect non-null array ") ;


    std::cout << " gs " << ( gs ? gs->sstr() : "-" ) << std::endl ;

    NPFold* fold = new NPFold ;
    fold->add( "igs", igs );
    fold->add( "gs", gs );
    fold->save("$FOLD");

    std::cout << "] SEvtTest::makeGenstepArrayFromVector \n";
    return 0 ;
}

int SEvtTest::saveExtra()
{
    SEvt::Create_ECPU() ;
    SEvt* evt = SEvt::Get_ECPU();
    evt->save();


    std::vector<std::pair<std::string, int>> kvs = {  {"red", 0}, {"green",1}, {"blue",2} } ;

    NP* somearray = NP::Make<int>(1) ;
    somearray->set_meta_kv(kvs);

    NP* otherarray = NP::Make<int>(1) ;
    otherarray->set_meta_kv(kvs);


    evt->saveExtra("somearray.npy", somearray );

    evt->saveExtra("$TMP", "otherarray.npy", otherarray );


    return 0 ;
}


int SEvtTest::Main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
    SEventConfig::SetRGModeTest();
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    int rc = 0 ;

    if(ALL||strcmp(TEST, "AddGenstep") == 0 )   rc += AddGenstep();
    if(ALL||strcmp(TEST, "LifeCycle") == 0 )    rc += LifeCycle();
    if(ALL||strcmp(TEST, "InputPhoton") == 0 )  rc += InputPhoton();
    if(ALL||strcmp(TEST, "getSaveDir") == 0 )   rc += getSaveDir();
    if(ALL||strcmp(TEST, "getDir") == 0 )       rc += getDir();
    if(ALL||strcmp(TEST, "setMetaProf") == 0 )  rc += setMetaProf();
    if(ALL||strcmp(TEST, "hostside_running_resize_") == 0 ) rc += hostside_running_resize_();
    if(ALL||strcmp(TEST, "CountNibbles") == 0 )  rc += CountNibbles();
    if(ALL||strcmp(TEST, "makeGenstepArrayFromVector") == 0 ) rc += makeGenstepArrayFromVector();
    if(ALL||strcmp(TEST, "saveExtra") == 0 )     rc += saveExtra();

    return rc ;
}

int main(int argc, char** argv){ return SEvtTest::Main(argc, argv); }

// ~/opticks/sysrap/tests/SEvtTest.sh
