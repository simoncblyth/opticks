#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"
#include "CSGClone.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* src = new CSGFoundry ; 
    src->makeDemoSolids(); 

    CSGFoundry* dst = CSGClone::Clone(src); 
    LOG(info) << " dst " << dst ;  

    int cf = CSGFoundry::Compare(src, dst); 
    assert( cf == 0 ); 

    return 0 ;  
}
