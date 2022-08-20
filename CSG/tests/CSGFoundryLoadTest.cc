#include "OPTICKS_LOG.hh"
#include "SSim.hh"
#include "stree.h"
#include "CSGFoundry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SSim::Create() ; 

    CSGFoundry* cf = CSGFoundry::Load() ; 
    LOG(info) << cf->desc() ; 

    stree* st = cf->sim->tree ; 
    LOG(info) << st->desc() ; 

    return 0 ; 
}
