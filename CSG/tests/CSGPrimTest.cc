#include "SSys.hh"
#include "scuda.h"
#include "Opticks.hh"
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

        float4 ce = pr->ce(); 

        std::cout 
            << " primIdx " << std::setw(10) << primIdx
            << " pr_primIdx " << std::setw(10) << pr_primIdx
            << " gasIdx " << std::setw(10) << gasIdx
            << " meshIdx " << std::setw(10) << meshIdx
            << " meshName " << std::setw(50) << meshName
            << " ce " 
            << "(" << std::setw(10) << std::fixed << std::setprecision(2) << ce.x 
            << "," << std::setw(10) << std::fixed << std::setprecision(2) << ce.y
            << "," << std::setw(10) << std::fixed << std::setprecision(2) << ce.z
            << "," << std::setw(10) << std::fixed << std::setprecision(2) << ce.w
            << ")" 
            << std::endl 
            ;
        //assert( pr_primIdx == primIdx ); 
    }


    return 0 ; 
}

