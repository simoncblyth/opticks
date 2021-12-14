#include "Opticks.hh"
#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGName.h"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* cfbase = ok.getFoundryBase("CFBASE") ; 
    LOG(info) << "cfbase " << cfbase ; 

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    LOG(info) << "foundry " << fd->desc() ; 
    fd->summary(); 

    const CSGName* id = fd->id ; 
    unsigned num_prim = fd->getNumPrim() ; 

    LOG(info) 
        << " id " << id
        << " num_prim " << num_prim 
        ;

    unsigned solidIdx = 0u ; 
    CSGPrimSpec psh = fd->getPrimSpec(solidIdx);   
   
    psh.dump();  


    return 0 ; 
}

