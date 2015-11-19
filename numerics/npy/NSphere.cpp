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

NPY<float>* NSphere::hemi_octahedron(unsigned int nsubdiv)
{
    NTrianglesNPY* hocta = NTrianglesNPY::hemi_octahedron();
    return hocta->subdivide(nsubdiv) ; 
}


NPY<float>* NSphere::hemi_octahedron(unsigned int nsubdiv, glm::mat4& m)
{
    NTrianglesNPY* ho = NTrianglesNPY::hemi_octahedron();
    NTrianglesNPY* tho = ho->transform(m);
    return tho->subdivide(nsubdiv) ; 
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



NPY<float>* NSphere::latlon(float zmin, float zmax, unsigned int npolar, unsigned int nazimuthal) 
{
    NTrianglesNPY* ll = NTrianglesNPY::sphere(zmin, zmax, npolar, nazimuthal);
    return ll->getBuffer();
}
