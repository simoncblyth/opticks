// TEST=NCSGListTest om-t

#include "OPTICKS_LOG.hh"
#include "NCSGList.hpp"
#include "NCSG.hpp"
#include "NPYBase.hpp"
#include "NPYList.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* csgpath = argc > 1 ? argv[1] : "$TMP/tboolean-box--" ; 
    unsigned verbosity = 0 ; 

    if(csgpath == NULL)
    {
        LOG(warning) << "Expecting 1st argument csgpath directory containing NCSG trees" ; 
        return 0 ;
    } 
   
    NCSGList* ls = NCSGList::Load(csgpath, verbosity );    
    if( ls == NULL )
    {
        LOG(warning) << "FAILED to load NCSG trees from " << csgpath  ; 
        return 0 ;
    }

    ls->dumpDesc();
    ls->dumpMeta();
    ls->dumpUniverse();

    unsigned num_trees = ls->getNumTrees(); 

    for(unsigned i=0 ; i < num_trees ; i++)
    {
        NCSG* tree = ls->getTree(i) ; 
        NPYList* npy = tree->getNPYList(); 
        LOG(info) << npy->desc() ; 

        NPY<float>* gt = tree->getGTransformBuffer(); 

        if(!gt) LOG(fatal) << "NO GTransformBuffer " ; 
        //assert(gt); 
    }




    return 0 ; 
}
