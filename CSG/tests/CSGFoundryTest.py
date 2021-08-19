#!/usr/bin/env python
import os, codecs, numpy as np

class Solid(object):
    def __init__(self, a, i):
        b = a.tobytes()
        self.i = i 
        self.label = a[i,0].tobytes().decode("utf-8")  # py3?
        self.numPrim = a[i,1]
        self.primOffset = a[i,2]
        self.extent = a[i,3].view(np.float32)

    def __repr__(self):
        return "Solid(%d) %10s numPrim:%3d primOffset:%4d extent:%10.4f " % ( self.i, self.label, self.numPrim, self.primOffset, self.extent )

class Foundry(object):
    def __init__(self, base):
        solid = np.load(os.path.join(base, "solid.npy"))
        prim = np.load(os.path.join(base, "prim.npy"))
        node = np.load(os.path.join(base, "node.npy"))
        plan = np.load(os.path.join(base, "plan.npy"))
        tran = np.load(os.path.join(base, "tran.npy"))

        self.solid = solid
        self.prim = prim
        self.node = node
        self.plan = plan
        self.tran = tran



if __name__ == '__main__':
    fd = Foundry("/tmp/FoundryTest_")




