// TEST=NCSGRoundTripTest om-t

#include "NPY.hpp"
#include "NConvexPolyhedron.hpp"

#include "OPTICKS_LOG.hh"
#include "NCSG.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    nconvexpolyhedron* cpol = nconvexpolyhedron::make_trapezoid_cube() ; 
    assert( cpol ) ; 
    cpol->dump_planes(); 

    cpol->verbosity = 3 ; 

    NCSG* a = NCSG::Adopt(cpol) ; 

    NPY<float>* buf = a->getPlaneBuffer();     

    LOG(info) << " PlaneBuffer " << buf->getShapeString() ; 

/*
    const char* treedir = "$TMP/NPY/NCSGRoundTripTest/nconvexpolyhedron" ;
    a->savesrc(treedir);     
*/


    return 0 ; 
}
