#include "OPTICKS_LOG.hh"
#include "CSGFoundry.h"
#include "CSGClone.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    char mode = argc > 1 ? argv[1][0] : 'K' ; 

    LOG(info) << " mode [" << mode << "]" ; 

    CSGFoundry* src = mode == 'D' ? CSGFoundry::MakeDemo() : CSGFoundry::Load() ; 

    // CSGFoundry::Load will load the geometry of the current OPTICKS_KEY unless CFBASE envvar override is defined  

    CSGFoundry* dst = CSGClone::Clone(src); 

    int cf = CSGFoundry::Compare(src, dst); 

    LOG(info) 
        << " src " << src 
        << " dst " << dst   
        << " cf " << cf 
        ;  

    assert( cf == 0 ); 

    return 0 ;  
}
