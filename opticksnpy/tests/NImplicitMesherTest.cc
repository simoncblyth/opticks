#include <iostream>
#include <glm/glm.hpp>

#include "NImplicitMesher.hpp"
#include "NTrianglesNPY.hpp"
#include "NSphere.hpp"
#include "NBox.hpp"
#include "NBBox.hpp"

#include "PLOG.hh"
#include "NPY_LOG.hh"


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    nsphere* sph = make_nsphere_ptr(0,0,0, 10) ;
    nbbox bb = sph->bbox();

    LOG(info) << argv[0] << " : " << bb.desc() ;

    int resolution = 100 ; 
    int verbosity = 1 ; 
    float bb_scale = 1.01 ; 

    NImplicitMesher im(resolution, verbosity, bb_scale);

    NTrianglesNPY* s_tris = im.sphere_test();
    assert(s_tris);

    NTrianglesNPY* tris = im(sph);
    assert(tris);


    return 0 ; 
}
