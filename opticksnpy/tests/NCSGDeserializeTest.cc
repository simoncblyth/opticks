/**
Tests directories of multiple trees::

    NCSGDeserializeTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-

**/

#include <iostream>

#include "BStr.hh"

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


 
void test_Deserialize(const char* basedir)
{
    int verbosity = 1 ; 
    std::vector<NCSG*> trees ; 
    int rc = NCSG::Deserialize(basedir, trees, verbosity );
    assert(rc == 0 );
}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    test_Deserialize( argc > 1 ? argv[1] : "$TMP/csg_py" );

    return 0 ; 
}


