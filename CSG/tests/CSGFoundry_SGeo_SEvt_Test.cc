
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sframe.h"

#include "SSys.hh"
#include "SEvt.hh"
#include "SOpticksResource.hh"

#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"

const char* FOLD = getenv("FOLD") ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    SEvt sev ;  
    sev.setReldir("ALL"); 
    sev.load(); 

    std::cout << sev.descDir() << std::endl ;  

    const char* loaddir = sev.getLoadDir(); 
    const char* cfbase = SOpticksResource::SearchCFBase(loaddir) ; 

    LOG(info)
         << " SearchCFBase " 
         << " loaddir " << loaddir 
         << " cfbase " << cfbase  
         ; 


    const CSGFoundry* fd = CSGFoundry::Load(cfbase);
    std::cout << fd->descBase() << std::endl ; 


    int ins_idx = SSys::getenvint("INS_IDX", 39216) ; 

    const SGeo* sg = (const SGeo*)fd ; 
    sframe fr ; 
    int rc = sg->getFrame(fr, ins_idx) ; 
    assert( rc == 0 ); 

    std::cout << "fr" << fr << std::endl ;   
    fr.save(FOLD); 
    fr.save_extras(FOLD); 
    fr.prepare(); 
 

    sev.setFrame(fr); 
    sev.setGeo(fd); 

    std::cout << sev.descComponent() ; 

    
    unsigned num_fold_photon = sev.getNumFoldPhoton(); 
    unsigned num_fold_hit    = sev.getNumFoldHit(); 
    unsigned num_print = std::min(100u, num_fold_photon) ; 

    std::cout 
        << " ins_idx " << ins_idx
        << " num_fold_photon " << num_fold_photon
        << " num_fold_hit    " << num_fold_hit
        << " num_print " << num_print 
        << std::endl 
        ; 

    std::cout << sev.descPhoton() << std::endl ; 
    std::cout << sev.descLocalPhoton() << std::endl ; 
    std::cout << sev.descFramePhoton() << std::endl ; 
    
    std::cout << fd->descBase() << std::endl ; 

   
    return 0 ; 
}

