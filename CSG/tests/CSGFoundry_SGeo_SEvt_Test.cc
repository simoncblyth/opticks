
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sframe.h"

#include "SSys.hh"
#include "SEvt.hh"

#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"

const char* FOLD = getenv("FOLD") ; 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const CSGFoundry* fd = CSGFoundry::Load();

    int ins_idx = SSys::getenvint("INS_IDX", 39216) ; 

    const SGeo* sg = (const SGeo*)fd ; 

    sframe fr ; 
    int rc = sg->getFrame(fr, ins_idx) ; 
    assert( rc == 0 ); 

    std::cout << "fr" << fr << std::endl ;   
    fr.save(FOLD); 
    fr.save_extras(FOLD); 
    fr.prepare(); 
 

    SEvt sev ;  
    sev.setReldir("ALL"); 
    sev.load(); 
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


    sphoton p ; 
    std::cout << "SEvt::getPhoton" << std::endl ; 
    for(unsigned idx=0 ; idx < num_print ; idx++)
    {
        sev.getPhoton(p, idx); 
        std::cout <<  p.desc() << std::endl  ; 
    }
    
    sphoton lp ; 
    std::cout << "SEvt::getLocalPhoton" << std::endl ; 
    for(unsigned idx=0 ; idx < num_print ; idx++)
    {
        sev.getLocalPhoton(lp, idx); 
        std::cout << lp.desc() << std::endl  ; 
    }

    sphoton fp ; 
    std::cout << "SEvt::getFramePhoton" << std::endl ; 
    for(unsigned idx=0 ; idx < num_print ; idx++)
    {
        sev.getFramePhoton(fp, idx); 
        std::cout << fp.desc() << std::endl  ; 
    }


    return 0 ; 
}

