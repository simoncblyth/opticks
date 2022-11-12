
#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sframe.h"

#include "SSim.hh"
#include "SEvt.hh"

#include "SPath.hh"
#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"


sframe test_getFrame_MOI(const CSGFoundry* fd)
{
    sframe fr = fd->getFrame() ;  // depends on MOI 

    std::cout << " fr " << std::endl << fr << std::endl ; 

    //const char* dir = SPath::Resolve("$TMP/CSG/CSGFoundry_getFrame_Test", DIRPATH); 
    //std::cout << dir << std::endl ; 
    //fr.save(dir) ; 

    return fr ; 
}

sframe test_getFrame_inst(const CSGFoundry* fd)
{
    int inst_idx = SSys::getenvint("INST", 0); 

    sframe fr ; 
    fd->getFrame(fr, inst_idx) ;  

    //const std::string& label = fd->getSolidLabel(sidx); 

    std::cout
        << " INST " << inst_idx << std::endl 
        << " fr " << std::endl 
        << fr << std::endl 
        << std::endl
        << "descInstance"
        << std::endl
        << fd->descInstance(inst_idx)
        << std::endl
        ; 

    //const char* dir = SPath::Resolve("$TMP/CSG/CSGFoundry_getFrame_Test", DIRPATH); 
    //std::cout << dir << std::endl ; 
    //fr.save(dir) ; 

    return fr ; 
}

const char* FOLD = getenv("FOLD") ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SSim::Create(); 
    const CSGFoundry* fd = CSGFoundry::Load();

    //sframe fr = test_getFrame_MOI(fd); 
    //sframe fr = test_getFrame_inst(fd); 

    const char* ipf_ = SEventConfig::InputPhotonFrame(); 
    const char* ipf = ipf_ ? ipf_ : "0" ; 
    std::cout << "ipf" << ipf << std::endl ;   

    sframe fr = fd->getFrame(ipf); 

    std::cout << "fr" << fr << std::endl ;   

    if( FOLD == nullptr )
    {
        LOG(error) << " define FOLD to save frame and other stuff " ; 
        return 0 ; 
    }

    fr.save(FOLD); 
    fr.save_extras(FOLD); 
    SEventConfig::SetRGModeSimulate(); 

    SEvt sev ;  

    NP* ip = sev.getInputPhoton_(); 

    NP* ipt0 = fr.transform_photon_m2w(ip, false)  ;
    NP* ipt1 = fr.transform_photon_m2w(ip, true )  ;

    if(ip == nullptr) return 0 ;  
 
    std::cout << " ip   " << ( ip  ? ip->sstr() : "-" )  << std::endl ; 
    std::cout << " ipt0  " << ( ipt0 ? ipt0->sstr() : "-" )  << std::endl ; 
    std::cout << " ipt1  " << ( ipt1 ? ipt1->sstr() : "-" )  << std::endl ; 


    ip->save(FOLD,  "ip.npy"); 
    ipt0->save(FOLD, "ipt0.npy"); 
    ipt1->save(FOLD, "ipt1.npy"); 

    sev.setFrame(fr); 

    NP* ipt2 = sev.getInputPhoton() ; 
    ipt2->save(FOLD, "ipt2.npy"); 


    return 0 ; 
}

