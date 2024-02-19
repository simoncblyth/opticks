/**
U4HitTest.cc
==============

::

    ~/o/u4/tests/U4HitTest.sh run_cat


**/

#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SProf.hh"
#include "spath.h"

#include "CSGFoundry.h"

#include "U4Hit.h"
#include "U4HitGet.h"


int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 

    LOG(info) ;  
    const char* rel = nullptr ; 
    int ins = 0 ;   // SEvt::EGPU
    int idx = 0 ;   // 1st valid index : A000 

    SEvt* sev = SEvt::LoadRelative(rel, ins, idx) ;   

    LOG(info) << "SEvt::LoadRelative sev " << ( sev ? "YES" : "NO " ) ; 

    LOG_IF(error, sev==nullptr ) << "SEvt::LoadRelative FAILED : ABORT " ; 
    if(sev == nullptr) return 0 ; 

    const char* cfbase = sev ? sev->getSearchCFBase() : nullptr ; 
    // search up dir tree starting from loaddir for dir with CSGFoundry/solid.npy

    LOG(info) << " cfbase " << ( cfbase ? cfbase : "-" ) ; 

    LOG_IF(error, cfbase==nullptr ) << "SEvt::getSearchCFBae FAILED : ABORT " ; 
    if(cfbase == nullptr) return 0 ; 

    LOG(info) << " cfbase " << ( cfbase ? cfbase : "-" ) ; 


    SSim::Create();  
    const CSGFoundry* fd = CSGFoundry::Load(cfbase);

    LOG(info) << " fd " << ( fd ? "YES" : "NO " ) ; 
    LOG_IF(error, fd==nullptr ) << " CSGFoundry::Load FAILED " ; 
    if(fd == nullptr) return 0 ; 

    sev->setGeo(fd); 
    
    
    assert( sev->hasInstance() );  // check that the instance is persisted and retrieved (via domain metadata)
    assert( sev == SEvt::Get(sev->instance) );  // check the loaded SEvt got slotted in expected slot 

    LOG(info) << "sev->descFull" << std::endl << sev->descFull() ; 

    unsigned num_hit = sev->getNumHit(); 

    LOG(info) << " num_hit " << num_hit ; 

    if(num_hit == 0) return 0 ; 


    for(unsigned hit_idx=0 ; hit_idx < num_hit ; hit_idx++ )
    {
        SProf::SetTag(hit_idx); 
        SProf::Add("Head"); 

        sphoton global = {} ; 
        sev->getHit(global, hit_idx); 

        sphit ht = {}  ; 
        sphoton local = {}  ; 
        sev->getLocalHit( ht, local,  hit_idx); 

        U4Hit hit = {} ; 
        U4HitGet::ConvertFromPhoton(hit,global,local, ht); 

        SProf::Add("Tail"); 

        int32_t drs = SProf::Delta_RS(); 

        if(drs > 0)
        {
            std::cout 
                << " hit_idx " << hit_idx 
                << " Delta_RS " << drs 
                << std::endl 
                ; 

            std::cout << " global " << global.desc() << std::endl ; 
            std::cout << " local " << local.desc() << std::endl ; 
            std::cout << " hit " << hit.desc() << std::endl ; 
            std::cout << " ht " << ht.desc() << std::endl ; 
        }

 

    }

    bool append = false ; 

    const char* _path = "$TMP/U4HitTest/U4HitTest.txt" ;  
    const char* path = spath::Resolve(_path) ; 
    SProf::Write(path, append); 

    /*
    U4Hit hit2 ; 
    U4HitGet::FromEvt(hit2, hit_idx, sev->instance ); 
    std::cout << " hit2 " << hit2.desc() << std::endl ; 
    */

  
    return 0 ; 
}

