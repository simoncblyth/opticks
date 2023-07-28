#!/usr/bin/env python
"""
CSGOptiXInstancesTest.py


     530 typedef struct OptixInstance
     531 {
     532     /// affine world-to-object transformation as 3x4 matrix in row-major layout
     533     float transform[12];
     534 
     535     /// Application supplied ID. The maximal ID can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID.
     536     unsigned int instanceId;
     537 
     538     /// SBT record offset.  Will only be used for instances of geometry acceleration structure (GAS) objects.
     539     /// Needs to be set to 0 for instances of instance acceleration structure (IAS) objects. The maximal SBT offset
     540     /// can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_SBT_OFFSET.
     541     unsigned int sbtOffset;
     542 
     543     /// Visibility mask. If rayMask & instanceMask == 0 the instance is culled. The number of available bits can be
     544     /// queried using OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK.
     545     unsigned int visibilityMask;
     546 
     547     /// Any combination of OptixInstanceFlags is allowed.
     548     unsigned int flags;
     549 
     550     /// Set with an OptixTraversableHandle.
     551     OptixTraversableHandle traversableHandle;
     552 
     553     /// round up to 80-byte, to ensure 16-byte alignment
     554     unsigned int pad[2];
     555 } OptixInstance;


::

    12 + 1 + 1 + 1 + 1 + 2 + 2 = 20 


In [16]: np.unique( traversableHandle, return_counts=True ) 
Out[16]: 
(array([140735712067594, 140735808176138, 140735808178698, 140735808181770, 140735808184842, 140735808187402, 140735808189962, 140735808192522, 140735808195082, 140735808207370], dtype=uint64),
 array([    1, 25600, 12615,  4997,  2400,   590,   590,   590,   590,   504]))


"""

import numpy as np
from opticks.ana.fold import Fold


class OptixInstance(object):
    def __init__(self, i):
        """
        :param i: instances
        """
        transform = i.view(np.float32)[:,:12].reshape(-1,3,4)
        instanceId = i[:,12]   
        sbtOffset = i[:,13]
        visibilityMask = i[:,14]
        flags = i[:,15]
        traversableHandle = i.view(np.uint64)[:,8]  # double size, so half idx  
        pad = i[:,18:20]
        assert( np.all(pad == 0 ) )

        self.i = i 

        self.transform = transform
        self.instanceId = instanceId 
        self.sbtOffset = sbtOffset  
        self.visibilityMask = visibilityMask
        self.flags = flags
        self.traversableHandle = traversableHandle
        self.pad = pad


if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))
    i = OptixInstance(f.instances)

    u_iid, n_iid = np.unique( i.instanceId, return_counts=True )  
    assert n_iid[1:].max() == 1  # should only be 1 of each instanceId other than 0  




pass
