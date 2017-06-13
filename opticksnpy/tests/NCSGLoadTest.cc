/**
Tests individual trees::

    NCSGLoadTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-/1

**/

#include <iostream>

#include "BFile.hh"
#include "BStr.hh"

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


 
void test_LoadTree(const char* treedir)
{
    int verbosity = 2 ;

    if(!BFile::ExistsDir(treedir))
    {
         LOG(warning) << "test_LoadTree no such dir " << treedir ;
         return ; 
    }
 
    NCSG* csg = NCSG::LoadTree(treedir, verbosity );
    assert(csg);
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    test_LoadTree( argc > 1 ? argv[1] : "$TMP/csg_py/1" );

    return 0 ; 
}


