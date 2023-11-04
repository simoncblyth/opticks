#include "OPTICKS_LOG.hh"
#include "ssys.h"
#include "SSim.hh"
#include "SBitSet.hh"
#include "CSGFoundry.h"
#include "CSGCopy.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    char mode = argc > 1 ? argv[1][0] : 'K' ; 

    LOG(info) << " mode [" << mode << "]" ; 

    SSim::Create(); 

    CSGFoundry* src = mode == 'D' ? CSGFoundry::MakeDemo() : CSGFoundry::Load_() ; 
    LOG_IF(fatal , src == nullptr ) << " NO GEOMETRY " ; 
    if(src == nullptr) return 1 ; 

    const SBitSet* elv = SBitSet::Create( src->getNumMeshName(), "ELV", "t" ); 

    LOG(info) << elv->desc() << std::endl << src->descELV(elv) ; 

    CSGFoundry* dst = CSGCopy::Select(src, elv ); 

    int cf = CSGFoundry::Compare(src, dst); 

    if(ssys::hasenv_("AFOLD")) src->save("$AFOLD"); 
    if(ssys::hasenv_("BFOLD")) dst->save("$BFOLD"); 

    LOG(info) 
        << " src " << src 
        << " dst " << dst   
        << " cf " << cf 
        ;  

    if( elv == nullptr || elv->all() )
    {
        LOG_IF(fatal, cf != 0 ) 
            << " UNEXPECTED DIFFERENCE " 
            << " DEBUG WITH :" 
            << std::endl 
            << " ~/opticks/CSG/tests/CSGCopyTest.sh ana "
            ;  

        assert( cf == 0 ); 
    }

    LOG(info) << " src.cfbase " << src->cfbase << " elv.spec " << elv->spec ; 
    if(src->cfbase && elv->spec)
    {
        LOG(info) << " src.cfbase " << src->cfbase << " elv.spec " << elv->spec ; 
    }

    return 0 ;  
}
