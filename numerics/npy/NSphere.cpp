#include "NSphere.hpp"
#include "NPY.hpp"

#include "icosahedron.hpp"

NPY<float>* NSphere::icosahedron(unsigned int nsubdiv)
{
    unsigned int ntris = icosahedron_ntris(nsubdiv);
    float* tris = icosahedron_tris(nsubdiv);

    NPY<float>* buf = NPY<float>::make( ntris, 3, 3); 
    buf->setData(tris);

    //buf->save("/tmp/icosahedron.npy");

    return buf ; 
}


