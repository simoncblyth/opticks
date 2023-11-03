#include "SSim.hh"
#include "CSGFoundry.h"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SSim::Create(); 

    CSGFoundry* fd = CSGFoundry::Load(); 
    LOG_IF(fatal, fd==nullptr) << CSGFoundry::LoadFailNotes() ; 
    if(fd == nullptr) return 1 ;  

    LOG(info) << "foundry " << fd->desc() ; 
    fd->summary(); 

    unsigned numPrim = fd->getNumPrim() ; 
    assert( fd->prim.size() == numPrim ); 

    for(unsigned primIdx=0 ; primIdx < std::min(10000u, numPrim) ; primIdx++)
    {
        const CSGPrim* pr = fd->getPrim(primIdx); 

        std::cout << std::endl ; 
        //std::cout << pr->desc() << std::endl ; 

        for(int nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset() + pr->numNode() ; nodeIdx++)
        {
            const CSGNode* nd = fd->getNode(nodeIdx); 
            std::cout << nd->desc()  ; 
        }
    }
    std::cout << std::endl ; 

    return 0 ; 
}

