#include "NSphere.hpp"
#include "NLog.hpp"
#include "NPY.hpp"

void test_icosahedron()
{
    NPY<float>* icos = NSphere::icosahedron(0) ; 
    icos->save("/tmp/icos.npy"); 
}

void test_latlon()
{
    NPY<float>* ll = NSphere::latlon() ; 
    ll->save("/tmp/ll.npy"); 
}


int main(int argc, char** argv)
{
    NLog nl("sphere.log","info");
    nl.configure(argc, argv, "/tmp");

    //test_icosahedron();
    test_latlon();

    return 0 ; 
}

/*
// python -c "import numpy as np; print np.load('/tmp/icos.npy')"

::

    In [1]: i = np.load("/tmp/icos.npy")

    In [2]: i
    Out[2]: 
    array([[[ 0.   ,  0.   ,  1.   ],
            [-0.526, -0.724,  0.447],
            [ 0.526, -0.724,  0.447]],
    ...
           [[ 0.851, -0.276, -0.447],
            [ 0.   , -0.894, -0.447],
            [ 0.   ,  0.   , -1.   ]]], dtype=float32)

    In [3]: np.linalg.norm(i.reshape(-1,3), 2, 1 )
    Out[3]: 
    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=float32)

    In [4]: i.shape
    Out[4]: (20, 3, 3)

*/


