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
from opticks.ana.blib import BLib
   

class Part(object):
    def __init__(self, part, trans, d):
        f = part.view(np.float32) 
        u = part.view(np.uint32)

        fc = part.copy().view(np.float32)
        fc[1][2] = 0  # scrub boundary in copy, as its known discrepant 

        tc = u[2][3]
        tcn = CSG_.desc(tc)
        bnd = u[1][2]

        # check the complements, viewing as float otherwise lost "-0. ok but not -0"
        comp = np.signbit(f[3,3])  
         # shift everything away, leaving just the signbit 
        comp2 = u[3,3] >> 31
        assert comp == comp2 

        # recover the gtransform index by getting rid of the complement signbit  
        gt = u[3][3] & 0x7fffffff


        self.f = f
        self.fc = fc

        self.tc = tc
        self.tcn = tcn 
        self.comp = comp
        self.gt = gt

        self.bnd = bnd
        self.bname = d.blib.bname(bnd)
        self.tran = trans[gt-1] if gt > 0 else np.eye(4)
        self.d = d 

    def __repr__(self):
        return "    Part %1s%2s %2s  %15s   %3d %25s   tz:%10.3f    %s  " % ( "!" if self.comp else " ", self.tc, self.gt, self.tcn, self.bnd, self.bname, self.tran[3][2], self.detail() )

    def maxdiff(self, other):
        return np.max( self.fc - other.fc )

    r = property(lambda self:self.f[0][3])

    r1co = property(lambda self:self.f[0][0])
    z1co = property(lambda self:self.f[0][1])
    r2co = property(lambda self:self.f[0][2])
    z2co = property(lambda self:self.f[0][3])

    z1 = property(lambda self:self.f[1][0])  # cy or zsphere
    z2 = property(lambda self:self.f[1][1])
    r1 = property(lambda self:self.f[0][2])
    r2 = property(lambda self:self.f[0][3])
    dz = property(lambda self:self.z2 - self.z1)
    dr = property(lambda self:self.r2 - self.r1)

    def detail(self):
        if self.tc == CSG_.ZSPHERE:
            msg = " r: %10.3f z1:%10.3f z2:%10.3f " % ( self.r, self.z1, self.z2 ) 
        elif self.tc == CSG_.SPHERE:
            msg = " r: %10.3f " % ( self.f[0][3]  ) 
        elif self.tc == CSG_.CYLINDER:
            msg = "   z1:%10.3f z2:%10.3f r :%10.3f " % ( self.z1, self.z2, self.r) 
        elif self.tc == CSG_.CONE:
            msg = "   z1:%10.3f z2:%10.3f r1:%10.3f r2:%10.3f " % ( self.z1co, self.z2co, self.r1co, self.r2co ) 
        else:
            msg = ""
        pass
        return msg 
        


class Prim(object):
    def __init__(self, primIdx, prim, d):
        self.primIdx = primIdx
        self.prim = prim 

        partOffset, numParts, tranOffset, planOffset = prim

        parts = d.part[partOffset:partOffset+numParts]
        trans = d.tran[tranOffset:,0]  # eg shape (2, 4, 4)

        self.parts = map(lambda _:Part(_,trans,d), parts)

        self.partOffset = partOffset
        self.numParts= numParts
        self.tranOffset = tranOffset
        self.planOffset = planOffset
        self.d = d

    def maxdiff(self, other):
        assert len(self.parts) == len(other.parts) 
        return max(map( lambda ab:ab[0].maxdiff(ab[1]), zip( self.parts, other.parts)))       
 
    def __repr__(self):
        return "primIdx %s partOffset %s numParts %s tranOffset %s planOffset %s  " % (self.primIdx, self.partOffset, self.numParts, self.tranOffset, self.planOffset )  

    def __str__(self):
        return "\n".join(["",repr(self)] + map(str,filter(lambda pt:pt.tc > 0, self.parts))) 


class Dir(object):
    def __init__(self, base):
        self.base = base
        self.blib = BLib.make(base)   # auto finds the idpath 
        self.prim = np.load(os.path.join(base, "primBuffer.npy"))
        self.part = np.load(os.path.join(base, "partBuffer.npy"))
        self.tran = np.load(os.path.join(base, "tranBuffer.npy"))
        self.prims = self.get_prims()

    def get_prims(self):
        pp = []
        for primIdx, prim in enumerate(self.prim):
            p = Prim(primIdx, prim, self)  
            pp.append(p)
        pass
        return pp

    def where_discrepant_prims(self, other, cut=0.1):
        assert len(self.prims) == len(other.prims) 
        return map(lambda i_ab:i_ab[0], filter(lambda i_ab:i_ab[1][0].maxdiff(i_ab[1][1]) > cut , enumerate(zip(self.prims, other.prims)) ))

    def get_discrepant_prims(self, other, cut=0.1):
        assert len(self.prims) == len(other.prims) 
        return filter(lambda i_ab:i_ab[1][0].maxdiff(i_ab[1][1]) > cut , enumerate(zip(self.prims, other.prims)) )

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


    


    

    


