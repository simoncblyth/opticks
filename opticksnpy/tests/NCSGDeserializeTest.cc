/**
Tests directories of multiple trees::

    NCSGDeserializeTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-

**/

#include <iostream>

#include "SSys.hh"
#include "BFile.hh"
#include "BStr.hh"

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


 
void test_Deserialize(const char* basedir, int verbosity)
{
    std::vector<NCSG*> trees ; 

    if(!BFile::ExistsDir(basedir))
    {
         LOG(warning) << "test_Deserialize no such dir " << basedir ;
         return ; 
    }

    int rc = NCSG::Deserialize(basedir, trees, verbosity );

    unsigned ntree = trees.size();
    for(unsigned i=0 ; i < ntree ; i++)
    {
        NCSG* tree = trees[i]; 
        nnode* root = tree->getRoot();
        LOG(info) << " root.desc : " << root->desc() ;

        NPY<float>* nodes = tree->getNodeBuffer();
        nodes->dump(); 

    }
    assert(rc == 0 );
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


