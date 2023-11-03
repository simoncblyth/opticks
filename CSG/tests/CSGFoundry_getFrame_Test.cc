
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sframe.h"
#include "ssys.h"
#include "SSim.hh"
#include "SEvt.hh"
#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"

const char* FOLD = getenv("FOLD") ; 


int test_InputPhoton(SEvt* evt, const sframe& fr )
{
    NP* ip = evt->getInputPhoton_(); 
    if(ip == nullptr) return 0 ;  

    NP* ipt0 = fr.transform_photon_m2w(ip, false)  ; // normalize:false
    NP* ipt1 = fr.transform_photon_m2w(ip, true )  ; // normalize:true 
 
    std::cout << " ip   " << ( ip  ? ip->sstr() : "-" )  << std::endl ; 
    std::cout << " ipt0  " << ( ipt0 ? ipt0->sstr() : "-" )  << std::endl ; 
    std::cout << " ipt1  " << ( ipt1 ? ipt1->sstr() : "-" )  << std::endl ; 

    ip->save(FOLD,  "ip.npy"); 
    ipt0->save(FOLD, "ipt0.npy"); 
    ipt1->save(FOLD, "ipt1.npy"); 

    evt->setFrame(fr); 
    NP* ipt2 = evt->getInputPhoton() ; 
    ipt2->save(FOLD, "ipt2.npy"); 

    return 0 ; 
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SSim::Create(); 
    const CSGFoundry* fd = CSGFoundry::Load();
    std::cout << " fd.brief " << fd->brief() << std::endl ;
    std::cout << " fd.desc  " << fd->desc() << std::endl ;
    std::cout << "[ fd->getFrameE " << std::endl ; 
    sframe fr = fd->getFrameE() ;  // via INST, MOI, OPTICKS_INPUT_PHOTON_FRAME "ipf"
    std::cout << "] fd->getFrameE " << std::endl ; 

    int INST = ssys::getenvint("INST",-1) ; 
    if(INST > -1) std::cout <<  "INST" << INST << std::endl << fd->descInstance(INST) << std::endl ; 

    std::cout << "[ fr " << std::endl << fr << std::endl << " ] fr " << std::endl ;   

    std::cout << " [ fr.save " << std::endl ; 
    fr.save(FOLD); 
    std::cout << " ] fr.save " << std::endl ; 
    std::cout << " [ fr.save_extras " << std::endl ; 
    fr.save_extras(FOLD); 
    std::cout << " ] fr.save_extras " << std::endl ; 

    SEvt* evt = SEvt::Create(0) ;  

    std::cout << " [ test_InputPhoton " << std::endl ; 
    int rc = test_InputPhoton(evt, fr ); 
    std::cout << " ] test_InputPhoton " << std::endl ; 

    return rc ; 
}

