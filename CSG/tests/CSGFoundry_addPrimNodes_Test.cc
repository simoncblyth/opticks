#include "OPTICKS_LOG.hh"

#include "scuda.h"
#include "saabb.h"
#include "SSim.hh"

#include "CSGFoundry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 


    std::vector<CSGNode> nds ; 
    nds.push_back( CSGNode::Sphere( 50.f )); 




    SSim::Create(); 

    CSGFoundry fd ;  

    unsigned num_prim = 1 ; 
    const char* label = "test" ; 
    fd.addSolid(num_prim, label ); 

    AABB bb ; 
    CSGPrim* pr = fd.addPrimNodes( bb, nds, nullptr ); 

    LOG(info) << fd.desc() ; 

    LOG(info) << pr->desc() ; 

    return 0 ; 
}
