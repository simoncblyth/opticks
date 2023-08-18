/**
CSGFoundry_CreateFromSimTest.cc
================================

1. loads SSim from $BASE/CSGFoundry
2. populates CSGFoundry with CSGFoundry::CreateFromSim
3. saves CSGFoundry to $FOLD (which should be differnent from $BASE)

**/

#include "OPTICKS_LOG.hh"
#include "SSim.hh"
#include "stree.h"
#include "CSGFoundry.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SSim* sim = SSim::Load("$BASE/CSGFoundry") ; 
    stree* st = sim->tree ; 
    std::cout << st->desc() ; 

    // CSGFoundry instanciation adopts SSim::INSTANCE loaded above 
    CSGFoundry* fd = CSGFoundry::CreateFromSim() ; 
    fd->save("$FOLD") ; 

    assert( fd->sim == sim ); 

    return 0 ;  
}



