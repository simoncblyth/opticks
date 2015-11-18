#include "NTesselate.hpp"
#include "NLog.hpp"
#include "NSphere.hpp"
#include "NPY.hpp"


int main(int argc, char** argv)
{
    NLog nl("tess.log","info");
    nl.configure(argc, argv);
    nl.init("/tmp");

    NPY<float>* icos = NSphere::icosahedron(0);
    icos->save("/tmp/icos.npy");

    NTesselate* tess = new NTesselate(icos);

    tess->subdivide(0);         

    NPY<float>* tris = tess->getTriangles();

    tris->save("/tmp/tris.npy");


    assert(tris->isEqualTo(icos));    


    return 0 ; 
}



