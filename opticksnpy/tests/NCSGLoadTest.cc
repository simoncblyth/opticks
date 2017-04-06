/**
Tests individual trees::

    NCSGLoadTest $TMP/tboolean-csg-two-box-minus-sphere-interlocked-py-/1

**/

#include <iostream>

#include "NPY.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"


struct NCSGLoadTest 
{
    NCSGLoadTest( const char* treedir, int index );
    NCSG csg ; 
};
  

NCSGLoadTest::NCSGLoadTest(const char* treedir, int index ) 
    : 
    csg(treedir, index)
{
    csg.load();
    csg.import();
    csg.dump("NCSGLoadTest");
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    NCSGLoadTest tst( argc > 1 ? argv[1] : "$TMP/csg_py/1" , 0 );

    return 0 ; 
}


