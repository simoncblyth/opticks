/**
Tests individual trees::

    NCSGLoadTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-/1


    NCSGLoadTest 0    
    NCSGLoadTest 66    
        # integer arguments are interpreted as lvidx and NCSG are 
        # loaded from the standard extras dir located within IDFOLD

**/

#include <iostream>

#include "SSys.hh"
#include "OpticksCSGMask.h"

#include "BFile.hh"
#include "BStr.hh"

#include "NPY.hpp"
#include "NCSGList.hpp"
#include "NCSG.hpp"
#include "NSceneConfig.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "OPTICKS_LOG.hh"


#include "SSys.hh"
#include "BOpticksResource.hh"



void test_coincidence( const std::vector<NCSG*>& trees )
{

    unsigned num_tree = trees.size() ;
    LOG(info) << " num_tree " << num_tree ; 


    unsigned count_coincidence(0);
    for(unsigned i=0 ; i < num_tree ; i++)
    {
        NCSG* csg = trees[i];
        if(num_tree == 1) csg->dump();

        unsigned num_coincidence = csg->get_num_coincidence();

        unsigned mask = csg->get_oper_mask();

        if(mask == CSGMASK_INTERSECTION ) 
        {
            LOG(info) << "skip intersection " ;             
            continue ; 
        }

        if(num_coincidence > 0) 
        {
            LOG(info)
                  << std::setw(40) 
                  << csg->get_soname()
                  << " " 
                  << csg->get_type_mask_string()
                  << csg->desc_coincidence() 
                 ;
            count_coincidence++ ; 
        }
    }

    LOG(info) 
          << " NCSGLoadTest trees " 
          << " num_tree " << num_tree 
          << " count_coincidence " << count_coincidence
          << " frac " << float(count_coincidence)/float(num_tree)
          ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    bool testgeo(false) ; 
    BOpticksResource okr(testgeo) ;  // no Opticks at this level 

    const char* basedir = okr.getDebuggingTreedir(argc, argv);

    LOG(info) << "basedir:" << basedir ; 

    if( basedir == NULL )
    {
        LOG(warning) << " this test operates beneath Opticks level, so it requires IDPATH envvar to locate the geocache " 
                     << " or provide directory path to persisted NCSG trees on commandline "  ;
        return 0 ; 
    }

    const char* gltfconfig = NULL ; 
    int verbosity = SSys::getenvint("VERBOSITY", 0);

    if(BFile::pathEndsWithInt(basedir))
    {
        std::vector<NCSG*> trees ;    
        NCSG* csg = NCSG::Load(basedir, gltfconfig);
        if(csg) trees.push_back(csg);   
        test_coincidence(trees);
    }
    else
    {
        // bool checkmaterial = false ; 
        // TODO: avoid this switch off, loading from the extras dir misinterprets a csg.txt as the boundaries 
        //       should change the names of the file (perhaps to extras.txt)

        NCSGList* ls = NCSGList::Load( basedir, verbosity );

        if(!ls)
        {
            LOG(fatal) << "this test requires the csg extras dir, which is created by running : op --gdml2gltf " ; 
            return 0 ;
        }
        const std::vector<NCSG*>& trees = ls->getTrees() ;    
        test_coincidence(trees);
    }

    return 0 ; 
}


