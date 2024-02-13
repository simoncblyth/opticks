#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "CSGFoundry.h"

#include "U4Hit.h"
#include "U4HitGet.h"



/*
struct U4HitTest
{
    U4HitTest(); 
};
*/




int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 

    LOG(info) ;  
    const char* rel = nullptr ; 
    int ins = 0 ;   // SEvt::EGPU
    int idx = 0 ;   // 1st valid index : A000 

    SEvt* sev = SEvt::LoadRelative(rel, ins, idx) ;   

    std::cout << " SEvt::LoadRelative sev " << ( sev ? "YES" : "NO " ) << std::endl ; 

    LOG_IF(error, sev==nullptr ) << "SEvt::LoadRelative FAILED : ABORT " ; 
    if(sev == nullptr) return 0 ; 

    const char* cfbase = sev ? sev->getSearchCFBase() : nullptr ; // search up dir tree starting from loaddir for dir with CSGFoundry/solid.npy

    std::cout << " cfbase " << ( cfbase ? cfbase : "-" ) << std::endl ; 

    LOG_IF(error, cfbase==nullptr ) << "SEvt::getSearchCFBae FAILED : ABORT " ; 
    if(cfbase == nullptr) return 0 ; 
    LOG(info) << " cfbase " << ( cfbase ? cfbase : "-" ) ; 


    SSim::Create();  
    const CSGFoundry* fd = CSGFoundry::Load(cfbase);

    std::cout << " fd " << ( fd ? "YES" : "NO " ) << std::endl ; 
    LOG_IF(error, fd==nullptr ) << " CSGFoundry::Load FAILED " ; 
    if(fd == nullptr) return 0 ; 

    sev->setGeo(fd); 
    
    
    assert( sev->hasInstance() );  // check that the instance is persisted and retrieved (via domain metadata)
    assert( sev == SEvt::Get(sev->instance) );  // check the loaded SEvt got slotted in expected slot 

    std::cout << "sev->descFull" << std::endl << sev->descFull() ; 

    unsigned num_hit = sev->getNumHit(); 

    LOG(info) << " num_hit " << num_hit ; 

    if(num_hit == 0) return 0 ; 


    unsigned hit_idx = 0 ; 
    sphoton global, local  ; 
    sev->getHit(global, hit_idx); 

    sphit ht ; 
    sev->getLocalHit( ht, local,  hit_idx); 


    U4Hit hit ; 
    U4HitGet::ConvertFromPhoton(hit,global,local, ht); 

    std::cout << " global " << global.desc() << std::endl ; 
    std::cout << " local " << local.desc() << std::endl ; 
    std::cout << " hit " << hit.desc() << std::endl ; 
    std::cout << " ht " << ht.desc() << std::endl ; 



    U4Hit hit2 ; 
    U4HitGet::FromEvt(hit2, hit_idx, sev->instance ); 
    std::cout << " hit2 " << hit2.desc() << std::endl ; 

  
    return 0 ; 
}

