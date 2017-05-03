/**
Tests directories of multiple trees::

    NCSGDeserializeTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-

**/

#include <iostream>

#include "SSys.hh"
#include "BStr.hh"

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


 
void test_Polygonize(const char* basedir, int verbosity, std::vector<NCSG*>& trees)
{
    int rc0 = NCSG::Deserialize(basedir, trees, verbosity );  // revive CSG node tree for each solid
    assert(rc0 == 0 );

    int rc1 = NCSG::Polygonize(basedir, trees, verbosity );
    assert(rc1 == 0 );
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

    std::vector<NCSG*> trees ; 
    test_Polygonize( argc > 1 ? argv[1] : "$TMP/csg_py", verbosity, trees);

    return 0 ; 
}


