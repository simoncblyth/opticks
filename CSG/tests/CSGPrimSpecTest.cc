
#include "scuda.h"
#include "SBitSet.hh"

#include "CSGFoundry.h"
#include "CSGCopy.h"
#include "CSGName.h"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    CSGFoundry* fdl = CSGFoundry::Load(); 
    LOG(info) << "foundry " << fdl->desc() ; 
    fdl->summary(); 

    SBitSet* elv = SBitSet::Create( fdl->getNumMeshName(), "ELV", nullptr ) ;

    if(elv)
    {
        LOG(info) << elv->desc() << std::endl << fdl->descELV(elv) ;
    }

    CSGFoundry* fd = CSGCopy::Select(fdl, elv); 

    const CSGName* id = fd->id ; 
    unsigned num_prim = fd->getNumPrim() ; 

    LOG(info) 
        << " id " << id
        << " num_prim " << num_prim 
        ;

    unsigned solidIdx = 0u ; 
    CSGPrimSpec psh = fd->getPrimSpec(solidIdx);   

    int modulo = 0 ; 
    psh.dump("", modulo);  

    return 0 ; 
}

