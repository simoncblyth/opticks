#include "IntersectSDF.hh"
#include "NPY.hpp"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* dir = argc > 1 ? argv[1] : "$TMP/UseOptiXGeometryInstancedOCtx" ;  

    float epsilon = 4e-4 ; 
    IntersectSDF is(dir, epsilon) ; 
    LOG(info) << is.desc() ; 

    unsigned rc = is.getRC() ; 
    LOG(info) << " rc " << rc ; 

    //assert( rc == 0 ); 
/*
    NPY<float>* tr = NPY<float>::make_identity_transforms(10); 
    tr->fillIndexFlat(); 
    IntersectSDF::FixColumnFour(tr);      
    tr->dump(); 
*/
    return 0 ; 
}

// om-;TEST=IntersectSDFTest om-t
