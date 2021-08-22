#!/usr/bin/env python
"""
::

    cx ; ipython -i tests/CSGOptiXSimulate.py


::

    314     unsigned instance_id = optixGetInstanceId() ;        // see IAS_Builder::Build and InstanceId.h 
    315     unsigned prim_id  = 1u + optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI 
    316     unsigned identity = (( prim_id & 0xff ) << 24 ) | ( instance_id & 0x00ffffff ) ;


    prim_id = i >> 24 
    prim_idx = ( i >> 24 ) - 1     ## index of the shape within the GAS 
    instance_id = i & 0x00ffffff



"""
import os, numpy as np
np.set_printoptions(suppress=True, edgeitems=5, linewidth=200,precision=3)

class CSGOptiXSimulate(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/CSGOptiX/CSGOptiXSimulate")
    def __init__(self):
        p = np.load(os.path.join(self.FOLD, "photons.npy"))
        globals()["p"] = p 


if __name__ == '__main__':
    cxs = CSGOptiXSimulate()

    print(p)

    n = p[:,3,:3]  # check normalization of the normal 
    nn = np.sum(n*n, axis=1)
    assert np.allclose( nn, 1. )


    i = p[:,3,3].view(np.uint32)
    ui,ui_counts = np.unique(i, return_counts=True)

    print(ui)
    print(ui_counts)

    prim_id = i >> 24 
    prim_idx = ( i >> 24 ) - 1 
    instance_id = i & 0x00ffffff



