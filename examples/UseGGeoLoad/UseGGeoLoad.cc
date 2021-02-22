#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "GGeo.hh"
#include "GParts.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
 
    Opticks ok(argc, argv);

    GGeo* gg = GGeo::Load(&ok);

    assert(gg); 

    unsigned nmm = gg->getNumMergedMesh(); 

    LOG(info) << " nmm " << nmm ; 

    gg->dumpParts(); 

    GParts* comp = gg->getCompositeParts(nmm-1); 

    assert( comp ); 

    unsigned num_sub = comp->getNumSubs(); 

    LOG(info) << " comp " << comp << " num_sub " << num_sub ; 

    for(unsigned i=0 ; i < num_sub ; i++)
    {
       GParts* sub = comp->getSub(i); 

       LOG(info) << " sub " << sub << " desc " << sub->desc() ; 

    }





    return 0 ; 
}


