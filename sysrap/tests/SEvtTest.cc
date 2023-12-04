// ~/opticks/sysrap/tests/SEvtTest.sh 

#include "OPTICKS_LOG.hh"
#include "OpticksGenstep.h"

#include "scuda.h"
#include "spath.h"
#include "sprof.h"
#include "stran.h"
#include "sdirectory.h"

#include "SEventConfig.hh"
#include "SEvt.hh"


struct SEvtTest
{
    static constexpr const int M = 1000000 ; 

    static void AddGenstep(); 
    static void LifeCycle(); 
    static void InputPhoton(); 
    static void getSaveDir(); 
    static void getOutputDir(); 
    static void setMetaProf(); 
    static void hostside_running_resize_(); 
    static void CountNibbles(); 

    static void Main(); 
 
};

void SEvtTest::AddGenstep()
{
   SEvt* evt = SEvt::Create(0) ; 
   assert( SEvt::Get(0) == evt );  

   for(unsigned i=0 ; i < 10 ; i++)
   {
       quad6 q ; 
       q.set_numphoton(1000) ; 
       unsigned gentype = i % 2 == 0 ? OpticksGenstep_SCINTILLATION : OpticksGenstep_CERENKOV ;  
       q.set_gentype(gentype); 

       SEvt::AddGenstep(q);    
   }

   std::cout << SEvt::Get(0)->desc() << std::endl ; 
}


void SEvtTest::LifeCycle()
{
    unsigned max_bounce = 9 ; 
    SEventConfig::SetMaxBounce(max_bounce); 
    SEventConfig::SetMaxRecord(max_bounce+1); 
    SEventConfig::SetMaxRec(max_bounce+1); 
    SEventConfig::SetMaxSeq(max_bounce+1); 

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
    assert( bounce1 == bounce0 + 1 ); 

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
}

void SEvtTest::InputPhoton()
{
    const char* ipf = SEventConfig::InputPhotonFrame();  
    if( ipf == nullptr) return ; 

    SEvt* evt = SEvt::Create(0) ; 
    LOG(info) << evt->desc() ;

    NP* ip = evt->getInputPhoton(); 
    LOG_IF(fatal, !ip ) << " FAILED TO getInputPhoton " ; 
    if(!ip) return ; 

    
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
}


/**
test_savedir
===============

savedir1 examples::

    /tmp/blyth/opticks/GEOM/SEvtTest/ALL
    /tmp/blyth/opticks/GEOM/Cheese/SEvtTest/ALL    # when GEOM envvar is Cheese  

Only after the save does the savedir get set.

**/


void SEvtTest::getSaveDir()
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
}


void SEvtTest::getOutputDir()
{
    SEvt* evt = SEvt::Create(0);
    const char* dir0 = evt->getOutputDir_OLD(); 
    const char* dir1 = evt->getOutputDir(); 

    LOG(info) 
        << "getOutputDir" << std::endl 
        << " dir0 [" << ( dir0 ? dir0 : "-" ) << "]" << std::endl 
        << " dir1 [" << ( dir1 ? dir1 : "-" ) << "]" << std::endl 
        ;
}


void SEvtTest::setMetaProf()
{
    SEvt* evt = SEvt::Create(0);

    sprof prof = {} ; 
    sprof::Stamp(prof); 
    evt->setMetaProf("test_setMeta", prof ); 

    std::cout << "evt->meta" << std::endl << evt->meta << std::endl ;  
}

/**
SEvtTest::hostside_running_resize_
------------------------------------

Profile time, VM, RSS change from hostside_running_resize

**/

void SEvtTest::hostside_running_resize_()
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

}

void SEvtTest::CountNibbles()
{
    const char* path = spath::Resolve("$SEQPATH"); 
    NP* seq = path ? NP::Load(path) : nullptr ;
    if(seq == nullptr) return ; 

    NP* nibcount = SEvt::CountNibbles(seq); 
    std::cout << nibcount->descTable<int>(7) ; 
}




void SEvtTest::Main()
{
    /*
    AddGenstep(); 
    LifeCycle(); 
    InputPhoton(); 
    getSaveDir(); 
    getOutputDir(); 
    setMetaProf(); 
    hostside_running_resize_() ; 
    */
    CountNibbles() ; 
    

}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    SEventConfig::SetRGModeTest(); 
    SEvtTest::Main(); 
    return 0 ; 
}
// ~/opticks/sysrap/tests/SEvtTest.sh 
