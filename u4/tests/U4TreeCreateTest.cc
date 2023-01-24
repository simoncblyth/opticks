#include "OPTICKS_LOG.hh"

#include "U4VolumeMaker.hh"
#include "U4Tree.h"
#include "stree.h"

const char* FOLD = getenv("FOLD"); 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const G4VPhysicalVolume* world = U4VolumeMaker::PV() ; 
    LOG_IF(error, world == nullptr) << " FAILED TO CREATE world with U4VolumeMaker::PV " ;   
    if(world == nullptr) return 0 ; 

    stree* st = new stree ; 
    st->level = 1 ;  

    U4Tree* tr = U4Tree::Create(st, world) ; 
    LOG(info) << tr->desc() ; 

    LOG(info) << " save stree to FOLD " << FOLD ; 
    st->save(FOLD); 

    return 0 ;  
}
