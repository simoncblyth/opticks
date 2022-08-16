#include "OPTICKS_LOG.hh"
#include "SEvt.hh"
#include "SSys.hh"
#include "CSGFoundry.h"

#include "U4Hit.h"
#include "U4HitGet.h"

int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 

    LOG(info) ;  
    SEvt* sev = SEvt::Load() ;   
    const char* cfbase = sev->getSearchCFBase() ; // search up dir tree starting from loaddir for dir with CSGFoundry/solid.npy
    const CSGFoundry* fd = CSGFoundry::Load(cfbase);
    sev->setGeo(fd); 

    std::cout << sev->descFull() ; 

    unsigned num_hit = sev->getNumHit(); 
    if(num_hit == 0) return 0 ; 


    unsigned idx = 0 ; 
    sphoton global, local  ; 
    sev->getHit(global, idx); 

    sphit ht ; 
    sev->getLocalHit( ht, local,  idx); 


    U4Hit hit ; 
    U4HitGet::ConvertFromPhoton(hit,global,local, ht); 

    std::cout << " global " << global.desc() << std::endl ; 
    std::cout << " local " << local.desc() << std::endl ; 
    std::cout << " hit " << hit.desc() << std::endl ; 
    std::cout << " ht " << ht.desc() << std::endl ; 


    U4Hit hit2 ; 
    U4HitGet::FromEvt(hit2, idx ); 
    std::cout << " hit2 " << hit2.desc() << std::endl ; 

  
    return 0 ; 
}

