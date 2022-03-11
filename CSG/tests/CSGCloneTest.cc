#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"
#include "CSGMaker.h"
#include "CSGClone.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    CSGFoundry* a = new CSGFoundry ; 
    a->maker->makeDemoSolids(); 

    CSGFoundry* b = CSGClone::Clone(a); 
    LOG(info) << " b " << b ;  

    int cf = CSGFoundry::Compare(a, b); 
    assert( cf == 0 ); 

    return 0 ;  
}
