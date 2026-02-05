/**
CSGFoundry_CreateFromSimTest.cc
================================

Creates CSGFoundry from SSim and SSim/stree

1. SSim::Load
2. populates CSGFoundry with CSGFoundry::CreateFromSim using
   the SSim that CSGFoundry instanciation adopts
3. saves CSGFoundry to $FOLD

**/

#include <csignal>
#include "OPTICKS_LOG.hh"
#include "SSim.hh"
#include "spath.h"
#include "stree.h"
#include "CSGFoundry.h"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    SSim* sim = SSim::Load() ;
    std::cout << "sim.tree.desc" << std::endl << sim->tree->desc() ;

    CSGFoundry* fd = CSGFoundry::CreateFromSim() ; // adopts SSim::INSTANCE
    fd->save("$FOLD") ;

    bool fd_expect = fd->sim == sim ;
    assert( fd_expect  );
    if(!fd_expect) std::raise(SIGINT);

    return 0 ;
}



