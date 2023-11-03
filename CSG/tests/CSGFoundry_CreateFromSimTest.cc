/**
CSGFoundry_CreateFromSimTest.cc
================================

Creates CSGFoundry from SSim and SSim/stree 

1. loads SSim from $HOME/.opticks/GEOM/$GEOM/CSGFoundry which must contain "SSim" subfold
2. populates CSGFoundry with CSGFoundry::CreateFromSim using 
   the SSim that CSGFoundry instanciation adopts
3. saves CSGFoundry to $FOLD (which should be different from $BASE)

**/

#include "OPTICKS_LOG.hh"
#include "SSim.hh"
#include "spath.h"
#include "stree.h"
#include "CSGFoundry.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* _base = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry" ; 
    const char* base = spath::Resolve(_base) ; 
    std::cout 
        << " _base " << _base << std::endl
        << " base " << ( base ? base : "-" ) << std::endl 
        ;

    SSim* sim = SSim::Load(base) ; 
    assert( sim && "$BASE folder needs to contain SSim subfold" ) ; 
    std::cout << "sim.tree.desc" << std::endl << sim->tree->desc() ; 


    CSGFoundry* fd = CSGFoundry::CreateFromSim() ; // adopts SSim::INSTANCE 
    //fd->save("$FOLD") ; 
    assert( fd->sim == sim ); 

    return 0 ;  
}



