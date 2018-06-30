#!/usr/bin/env python
"""
prim.py 
=========

Loads a list of primitives from a GParts persisted directory.
Used for debugging things such as missing transforms.

See: notes/issues/OKX4Test_partBuffer_difference.rst

TODO: consolidate ana/prim.py with dev/csg/GParts.py 

"""
import sys, os, numpy as np

from opticks.sysrap.OpticksCSG import CSG_

class Part(object):
    def __init__(self, part, trans):
        f = part.view(np.float32) 
        u = part.view(np.uint32)
        tc = u[2][3]
        tcn = CSG_.desc(tc)
        gt = u[3][3]
        self.f = f
        self.tc = tc
        self.tcn = tcn 
        self.gt = gt
        self.tran = trans[gt-1] if gt > 0 else np.eye(4)

    def __repr__(self):
        return "    Part %2s %2s  %15s     tz:%10.3f    %s  " % ( self.tc, self.gt, self.tcn, self.tran[3][2], self.detail() )

    def detail(self):
        if self.tc == CSG_.ZSPHERE:
            msg = " r: %10.3f z1:%10.3f z2:%10.3f " % ( self.f[0][3], self.f[1][0], self.f[1][1] ) 
        elif self.tc == CSG_.SPHERE:
            msg = " r: %10.3f " % ( self.f[0][3]  ) 
        elif self.tc == CSG_.CYLINDER:
            msg = " r: %10.3f z1:%10.3f z2:%10.3f " % ( self.f[0][3], self.f[1][0], self.f[1][1] ) 
        else:
            msg = ""
        pass
        return msg 
        


class Prim(object):
    def __init__(self, primIdx, prim, d):
        self.primIdx = primIdx
        self.prim = prim 

        partOffset = prim[0]
        numParts = prim[1]
        tranOffset = prim[2]
        planOffset = prim[3]

        parts = d.part[partOffset:partOffset+numParts]
        trans = d.tran[tranOffset:,0]  # eg shape (2, 4, 4)

        self.parts = map(lambda _:Part(_,trans), parts)

        self.partOffset = partOffset
        self.numParts= numParts
        self.tranOffset = tranOffset
        self.planOffset = planOffset

        self.d = d

    def __repr__(self):
        return "primIdx %s prim %s partOffset %s numParts %s tranOffset %s planOffset %s  " % (self.primIdx, repr(self.prim), self.partOffset, self.numParts, self.tranOffset, self.planOffset )  

    def __str__(self):
        return "\n".join(["",repr(self)] + map(str,filter(lambda pt:pt.tc > 0, self.parts))) 


class Dir(object):
    def __init__(self, base):
        self.base = base
        self.prim = np.load(os.path.join(base, "primBuffer.npy"))
        self.part = np.load(os.path.join(base, "partBuffer.npy"))
        self.tran = np.load(os.path.join(base, "tranBuffer.npy"))

    def prims(self):
        pp = []
        for primIdx, prim in enumerate(self.prim):
            p = Prim(primIdx, prim, self)  
            pp.append(p)
        pass
        return pp

    def __repr__(self):
        return "\n".join([self.base,"prim %s part %s tran %s " % ( repr(self.prim.shape), repr(self.part.shape), repr(self.tran.shape))])
       

if __name__ == '__main__':

    ddir = "/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103/GPartsAnalytic/5"
    dir_ = sys.argv[1] if len(sys.argv) > 1 else ddir

    d = Dir(dir_)
    print d

    pp = d.prims()
    #assert len(pp) == 5

    for p in pp:
        print p
    pass

    #print d.tran


    


    

    


