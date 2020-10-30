#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "SphereOfTransforms.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    float radius = 1000.f ; 
    unsigned num_theta = 5 ;  // odd to get one to land on equator
    unsigned num_phi   = 8 ; 

    SphereOfTransforms sot(radius, num_theta, num_phi); 
    LOG(info) << sot.desc(); 

    NPY<float>* tr = sot.getTransforms(); 
    tr->dump(); 

    const char* path = "$TMP/optickscore/tests/SphereOfTransformsTest/tr.npy" ; 
    LOG(info) << path ;  
    tr->save(path); 

    return 0 ; 
}
