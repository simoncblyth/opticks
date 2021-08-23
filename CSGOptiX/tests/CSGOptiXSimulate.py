#!/usr/bin/env python
"""
::

    cx ; ipython -i tests/CSGOptiXSimulate.py



__closesthit__ch::

    331     unsigned instance_idx = optixGetInstanceId() ;    // see IAS_Builder::Build and InstanceId.h 
    332     unsigned prim_idx  = optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI_11N  (1+index-of-CSGPrim within CSGSolid/GAS)
    333     unsigned identity = (( prim_idx & 0xffff ) << 16 ) | ( instance_idx & 0xffff ) ;

    prim_idx = ( i >> 16 )      ## index of bbox within within the GAS 
    instance_idx = i & 0xffff   ## flat 

NB getting zero for the flat instance_idx (single IAS, all transforms in it) 
**DOES** tell you that its a global intersect 

Now how to lookup what a prim_id corresponds to ?
Currently the only names CSGFoundry holds are mesh names


In [2]: prim_idx
Out[2]: array([19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19], dtype=uint32)

In [3]: instance_id
Out[3]: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint32)


"""
import os, numpy as np
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

import matplotlib.pyplot as plt




class CSGOptiXSimulate(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/CSGOptiX/CSGOptiXSimulate")
    def __init__(self):
        p = np.load(os.path.join(self.FOLD, "photons.npy"))
        globals()["p"] = p 


if __name__ == '__main__':
    cxs = CSGOptiXSimulate()

    print(p)

    #n = p[:,3,:3]  # check normalization of the normal 
    #nn = np.sum(n*n, axis=1)
    #assert np.allclose( nn, 1. )


    i = p[:,3,3].view(np.uint32)
    ui,ui_counts = np.unique(i, return_counts=True)

    print(ui)
    print(ui_counts)

    prim_idx = ( i >> 16 ) 
    instance_id = i & 0xffff
 
    print("prim_idx")
    print(prim_idx)
    print("instance_id")
    print(instance_id)

    boundary = p[:,2,3].view(np.uint32)

    print("boundary")
    print(boundary)


    fig, ax = plt.subplots()
    ax.scatter( p[:,0,0], p[:,0,1] )
    fig.show()

    




