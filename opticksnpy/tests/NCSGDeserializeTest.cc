/**
Tests directories of multiple trees::

    NCSGDeserializeTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-

**/

#include <iostream>

#include "SSys.hh"
#include "BFile.hh"
#include "BStr.hh"

#include "NPY.hpp"
#include "NCSGList.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


 
void test_Deserialize(const char* basedir, int verbosity)
{

     NCSGList* ls = NCSGList::Load(basedir, verbosity );
     if( ls == NULL )
     {
          LOG(warning) << "failed to NCSGList::Load from " << basedir ; 
          return ; 
     }

    unsigned ntree = ls->getNumTrees();
    for(unsigned i=0 ; i < ntree ; i++)
    {
        NCSG* tree = ls->getTree(i); 
        nnode* root = tree->getRoot();
        LOG(info) << " root.desc : " << root->desc() ;

        NPY<float>* nodes = tree->getNodeBuffer();
        nodes->dump(); 

    }

}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    int verbosity = SSys::getenvint("VERBOSITY", 0 );
    LOG(info) << " argc " << argc 
              << " argv[0] " << argv[0] 
              << " VERBOSITY " << verbosity 
              ;  

    test_Deserialize( argc > 1 ? argv[1] : "$TMP/csg_py", verbosity);

    return 0 ; 
}


