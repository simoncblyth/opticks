#include <iostream>

#include "NPY.hpp"
#include "NCSG.hpp"

#include "NPolygonizer.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    const char* treedir = argc > 1 ? argv[1] : "$TMP/tboolean-hyctrl--/1" ;

    int verbosity = 2 ; 

    NCSG* csg = NCSG::LoadTree(treedir, verbosity );

    assert( csg );

    NPolygonizer poly(csg);

    NTrianglesNPY* tris = poly.polygonize();

    assert(tris);


    return 0 ; 
}


