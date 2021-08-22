#include "SSys.hh"
#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGName.h"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* fd = CSGFoundry::Load(SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" ), "CSGFoundry"); 
    LOG(info) << "foundry " << fd->desc() ; 
    fd->summary(); 

    const CSGName* id = fd->id ; 
    unsigned num_prim = fd->getNumPrim() ; 

    unsigned solidIdx = 0u ; 
    CSGPrimSpec psh = fd->getPrimSpec(solidIdx);   
   
    psh.dump();  




    return 0 ; 
}

