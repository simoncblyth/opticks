
//#include "scuda.h"
#include "CSGFoundry.h"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* fd = CSGFoundry::Load(); 
    LOG(info) << "foundry " << fd->desc() ; 
    fd->summary(); 

    unsigned num_prim = fd->getNumPrim() ; 

    LOG(info) 
        << " num_prim " << num_prim 
        ;

    unsigned solidIdx = 0u ; 
    CSGPrimSpec psh = fd->getPrimSpec(solidIdx);   

    int modulo = 0 ; 
    psh.dump("", modulo);  

    return 0 ; 
}

