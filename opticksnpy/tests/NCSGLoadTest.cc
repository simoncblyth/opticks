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

#include "NPY_LOG.hh"
#include "PLOG.hh"


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
                  << csg->soname()
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
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    BOpticksResource okr ;  // no Opticks at this level 

    const char* basedir = okr.getDebuggingTreedir(argc, argv);
    const char* gltfconfig = NULL ; 
    int verbosity = SSys::getenvint("VERBOSITY", 0);

    if(BFile::pathEndsWithInt(basedir))
    {
        std::vector<NCSG*> trees ;    
        NCSG* csg = NCSG::LoadCSG(basedir, gltfconfig);
        if(csg) trees.push_back(csg);   
        test_coincidence(trees);
    }
    else
    {
        NCSGList* ls = NCSGList::Load( basedir, verbosity );
        const std::vector<NCSG*>& trees = ls->getTrees() ;    
        test_coincidence(trees);
    }

    return 0 ; 
}


