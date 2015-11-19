#include "NSphere.hpp"
#include "NTrianglesNPY.hpp"


NPY<float>* NSphere::icosahedron(unsigned int nsubdiv)
{
    NTrianglesNPY* icos = NTrianglesNPY::icosahedron();
    return icos->subdivide(nsubdiv) ; 
}
NPY<float>* NSphere::octahedron(unsigned int nsubdiv)
{
    NTrianglesNPY* octa = NTrianglesNPY::octahedron();
    return octa->subdivide(nsubdiv) ; 
}
NPY<float>* NSphere::cube(unsigned int nsubdiv)
{
    NTrianglesNPY* cube = NTrianglesNPY::cube();
    return cube->subdivide(nsubdiv) ; 
}


NPY<float>* NSphere::latlon(unsigned int npolar, unsigned int nazimuthal) 
{
    NTrianglesNPY* ll = NTrianglesNPY::sphere(npolar, nazimuthal);
    return ll->getBuffer();
}



