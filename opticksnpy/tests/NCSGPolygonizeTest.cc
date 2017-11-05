/**
Tests directories of multiple trees::

    NCSGDeserializeTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-

**/

#include <iostream>

#include "SSys.hh"
#include "BStr.hh"
#include "BFile.hh"

#include "NPY.hpp"
#include "NCSGList.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


 
void test_Polygonize(const char* basedir, int verbosity )
{
    NCSGList* ls = NCSGList::Load(basedir, verbosity );
    if(!ls)
    {
         LOG(warning) << "test_Polygonize no such dir " << basedir ;
         return ; 
    }
    int rc1 = ls->polygonize();
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

    test_Polygonize( argc > 1 ? argv[1] : "$TMP/csg_py", verbosity );

    return 0 ; 
}


