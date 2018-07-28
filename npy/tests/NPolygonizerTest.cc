#include "NPY.hpp"
#include "NCSG.hpp"
#include "NPolygonizer.hpp"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << " argc " << argc << " argv[0] " << argv[0] ;  

    const char* treedir = argc > 1 ? argv[1] : "$TMP/tboolean-hyctrl--/1" ;

    const char* config = "csg_bbox_parsurf=1" ;

    NCSG* csg = NCSG::Load(treedir, config );

    if(!csg)
    {
        LOG(fatal) << "NO treedir/tree " << treedir ; 
        return 0 ;  
    }

    NPolygonizer poly(csg);

    NTrianglesNPY* tris = poly.polygonize();

    assert(tris);

    return 0 ; 
}


