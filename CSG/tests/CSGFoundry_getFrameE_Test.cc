#include "scuda.h"
#include "squad.h"
#include "sqat4.h"
#include "sframe.h"
#include "ssys.h"
#include "SSim.hh"
#include "SEvt.hh"
#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"

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

    std::cout << " [ fr.save " << std::endl ; 
    fr.save("$FOLD"); 
    std::cout << " ] fr.save " << std::endl ; 
    std::cout << " [ fr.save_extras " << std::endl ; 
    fr.save_extras("$FOLD"); 
    std::cout << " ] fr.save_extras " << std::endl ; 


    std::cout << fr << std::endl ; 


    return 0 ; 
}

