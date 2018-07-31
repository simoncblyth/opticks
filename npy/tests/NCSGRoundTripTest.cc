// TEST=NCSGRoundTripTest om-t

#include "NPY.hpp"
#include "NConvexPolyhedron.hpp"

#include "OPTICKS_LOG.hh"
#include "NCSG.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    nconvexpolyhedron* a_cpol = nconvexpolyhedron::make_trapezoid_cube() ; 
    assert( a_cpol ) ; 
    a_cpol->dump_planes(); 
    a_cpol->dump_srcvertsfaces(); 

    

    a_cpol->verbosity = 2 ; 

    LOG(error) << " before Adopt " ; 
    NCSG* a = NCSG::Adopt(a_cpol) ; 
    LOG(error) << " after Adopt " ; 

    NPY<float>* a_planes = a->getPlaneBuffer();     
    LOG(info) << " a_planes " << a_planes->getShapeString() ; 

    const char* treedir = "$TMP/NPY/NCSGRoundTripTest/nconvexpolyhedron/1" ;
    a->savesrc(treedir);     

    NCSG* b = NCSG::Load(treedir) ;     

    NPY<float>* b_planes = b->getPlaneBuffer();     
    LOG(info) << " b_planes " << b_planes->getShapeString() ; 


    nconvexpolyhedron* b_cpol = dynamic_cast<nconvexpolyhedron*>(b->getRoot());     
    assert( b_cpol ) ; 
    b_cpol->dump_planes();  
    b_cpol->dump_srcvertsfaces(); 


    return 0 ; 
}
