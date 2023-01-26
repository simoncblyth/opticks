#include "OPTICKS_LOG.hh"

#include "snd.h"
std::vector<snd> snd::node  = {} ; 
std::vector<spa> snd::param = {} ; 
std::vector<sxf> snd::xform = {} ; 
std::vector<sbb> snd::aabb  = {} ; 
// HMM: how to avoid ? 


#include "U4VolumeMaker.hh"
#include "U4Tree.h"
#include "stree.h"

const char* FOLD = getenv("FOLD"); 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    stree* st = new stree ; 
    st->level = 1 ;  

    if( argc > 1 )
    {
        LOG(info) << " load stree from FOLD " << FOLD ; 
        st->load(FOLD); 
    }
    else
    {
        const G4VPhysicalVolume* world = U4VolumeMaker::PV() ; 
        LOG_IF(error, world == nullptr) << " FAILED TO CREATE world with U4VolumeMaker::PV " ;   
        if(world == nullptr) return 0 ; 

        U4Tree* tr = U4Tree::Create(st, world) ; 
        LOG(info) << tr->desc() ; 

        LOG(info) << " save stree to FOLD " << FOLD ; 
        st->save(FOLD); 
    }

    std::cout << st->desc() ; 


    return 0 ;  
}
