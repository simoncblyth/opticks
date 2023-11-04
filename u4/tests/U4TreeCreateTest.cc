#include "OPTICKS_LOG.hh"
#include "spath.h"
#include "stree.h"

#include "U4VolumeMaker.hh"
#include "U4Tree.h"

const char* FOLD = spath::Resolve("/tmp/$USER/opticks/U4TreeCreateTest"); 

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    stree* st = new stree ; 
    st->level = 1 ;  

    if( argc > 1 )
    {
        LOG(info) << " load stree from FOLD " << FOLD ; 
        int rc = st->load(FOLD); 
        if(rc != 0) return rc ; 
    }
    else
    {
        const G4VPhysicalVolume* world = U4VolumeMaker::PV() ; 
        LOG_IF(error, world == nullptr) << " FAILED TO CREATE world with U4VolumeMaker::PV " ;   
        if(world == nullptr) return 0 ; 

        U4Tree* tr = U4Tree::Create(st, world) ; 
        assert( tr ); 
        //LOG(info) << tr->desc() ; 

        LOG(info) << " save stree to FOLD " << FOLD ; 
        st->save(FOLD); 
    }

    std::cout << st->desc() ; 
    return 0 ;  
}
