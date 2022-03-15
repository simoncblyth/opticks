#include "SSys.hh"
#include "scuda.h"
#include "CSGFoundry.h"
#include "CSGName.h"

#include "OPTICKS_LOG.hh"

std::string DescPrim(const CSGFoundry* fd, unsigned primIdx )
{
    const CSGName* id = fd->id ; 
    const CSGPrim* pr = fd->getPrim(primIdx); 
    unsigned gasIdx = pr->repeatIdx(); 
    unsigned meshIdx = pr->meshIdx(); 
    unsigned pr_primIdx = pr->primIdx(); 
    const char* meshName = id->getName(meshIdx);

    int numNode = pr->numNode() ; 
    int nodeOffset = pr->nodeOffset() ; 

    float4 ce = pr->ce(); 

    std::stringstream ss ; 
    ss 
        << " primIdx " << std::setw(5) << primIdx
        << " pr_primIdx " << std::setw(5) << pr_primIdx
        << " gasIdx " << std::setw(2) << gasIdx
        << " meshIdx " << std::setw(3) << meshIdx
        << " meshName " << std::setw(15) << meshName
        << " ce " 
        << "(" << std::setw(10) << std::fixed << std::setprecision(2) << ce.x 
        << "," << std::setw(10) << std::fixed << std::setprecision(2) << ce.y
        << "," << std::setw(10) << std::fixed << std::setprecision(2) << ce.z
        << "," << std::setw(10) << std::fixed << std::setprecision(2) << ce.w
        << ")" 
        << " numNode " << std::setw(3) << numNode 
        << " nodeOffset " << std::setw(6) << nodeOffset 
        ;

    std::string s = ss.str();
    return s ; 
} 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* fd = CSGFoundry::Load(); 
 
    LOG(info) << "foundry " << fd->desc() ; 
    fd->summary(); 

    unsigned numPrim = fd->getNumPrim() ; 
    assert( fd->prim.size() == numPrim ); 

    for(unsigned primIdx=0 ; primIdx < std::min(10000u, numPrim) ; primIdx++)
    {
        std::cout << DescPrim(fd, primIdx) << std::endl ; 
        //const CSGPrim* pr = fd->getPrim(primIdx); 
        //std::cout << pr->desc() << std::endl ; 
        //for(int nodeIdx=nodeOffset ; nodeIdx < nodeOffset + numNode ; nodeIdx++)

    }

    return 0 ; 
}

