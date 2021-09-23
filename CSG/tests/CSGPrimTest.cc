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

    std::cout 
        << " fd->prim.size() " << fd->prim.size() 
        << " fd->getNumPrim() " << num_prim
        << std::endl 
        ; 

    for(unsigned i=0 ; i < std::min(10000u, num_prim) ; i++)
    {
        unsigned primIdx = i ; 
        const CSGPrim* pr = fd->getPrim(primIdx); 

        unsigned gasIdx = pr->repeatIdx(); 
        unsigned meshIdx = pr->meshIdx(); 
        unsigned pr_primIdx = pr->primIdx(); 
        const char* meshName = id->getName(meshIdx);

        std::cout 
            << " primIdx " << std::setw(10) << primIdx
            << " pr_primIdx " << std::setw(10) << pr_primIdx
            << " gasIdx " << std::setw(10) << gasIdx
            << " meshIdx " << std::setw(10) << meshIdx
            << " meshName " << meshName
            << std::endl 
            ;
        //assert( pr_primIdx == primIdx ); 
    }


    return 0 ; 
}
