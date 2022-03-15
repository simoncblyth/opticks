#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SBitSet.hh"
#include "CSGFoundry.h"
#include "CSGCopy.h"





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    char mode = argc > 1 ? argv[1][0] : 'K' ; 

    LOG(info) << " mode [" << mode << "]" ; 


    CSGFoundry* src = mode == 'D' ? CSGFoundry::MakeDemo() : CSGFoundry::Load() ; 
    // CSGFoundry::Load the geometry of the current OPTICKS_KEY unless CFBASE envvar override is defined  

    const SBitSet* elv = SBitSet::Create( src->getNumMeshName(), "ELV", "t" ); 

    LOG(info) << elv->desc() << std::endl << src->descELV(elv) ; 

    CSGFoundry* dst = CSGCopy::Select(src, elv ); 

    int cf = CSGFoundry::Compare(src, dst); 

    LOG(info) 
        << " src " << src 
        << " dst " << dst   
        << " cf " << cf 
        ;  

    if( elv == nullptr || elv->all() )
    {
        assert( cf == 0 ); 
    }

    LOG(info) << " src.cfbase " << src->cfbase << " elv.spec " << elv->spec ; 
    if(src->cfbase && elv->spec)
    {
        LOG(info) << " src.cfbase " << src->cfbase << " elv.spec " << elv->spec ; 
    }


    return 0 ;  
}
