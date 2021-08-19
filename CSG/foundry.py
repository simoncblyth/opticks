#!/usr/bin/env python

import os, sys, logging, codecs, numpy as np
from opticks.sysrap.OpticksCSG import CSG_ as CSG 
from complete_binary_tree import layout_tree


log = logging.getLogger(__name__)

class BB(object):
    """

                +--------+       
                |        | 
      z1  +--------+     | 
          |     |  |     |
          |     +--|-----+   y1  
          |        |
      z0  +--------+     y0
         x0        x1
    """
    def __init__(self, bb=[-0.,-0.,-0.,0.,0.,0.]):
        self.x0 = bb[0]
        self.y0 = bb[1]
        self.z0 = bb[2]
        self.x1 = bb[3]
        self.y1 = bb[4]
        self.z1 = bb[5]
        self.data = bb 

    def __repr__(self):
        fmt = "%10.3f "
        fmt = fmt * 6 
        return fmt % tuple(self.data)

    def include(self, other):
        if other.x0 < self.x0: self.x0 = other.x0
        if other.y0 < self.y0: self.y0 = other.y0
        if other.z0 < self.z0: self.z0 = other.z0

        if other.x1 > self.x1: self.x1 = other.x1
        if other.y1 > self.y1: self.y1 = other.y1
        if other.z1 > self.z1: self.z1 = other.z1



class Tran(object):
    def __init__(self, idx, tr, it):
        self.idx = idx
        self.tr = tr 
        self.it = it 
    def __repr__(self):
        return "\n".join([str(self.idx), repr(self.tr)])

class Node(object):
    """

                                    1
                      10                          11
                100       101               110         111
             1000 1001 1100 1101        1100 1101     1110 1111


    """
    def __init__(self, item, fd):
        self.fd = fd 
        self.tc = item[3,2].view(np.int32)
        self.tyd = CSG.desc(self.tc)
        self.ty = self.tyd[:2]
        self.idx = item[1,3].view(np.int32)
        self.depth = len("{0:b}".format(self.idx+1)) - 1   # complete binary trees are very special  
        self.tref = item[3,3].view(np.int32) & 0x7fffffff 
        self.tidx = self.tref - 1
        self.cm = bool(item[3,3].view(np.uint32) & 0x80000000 >> 31) 
        self.bb = BB(item[2:4].ravel()[:6])
        self.pa = item[0:2].ravel()[:6]
        self.tr = self.fd.tran[self.tidx] if self.tref > 0 else None
        self.it = self.fd.itra[self.tidx] if self.tref > 0 else None
        self.tran = Tran(self.tidx, self.tr, self.it) if self.tidx > -1 else None
        self.scm = "!" if self.cm else ":"
        self.label = "%s%s%s" % (1+self.idx,self.scm, self.ty) 

    def __repr__(self):
        return "Node %2d%s%2s tidx:%5d cm:%1d bb %30s pa %30s " % (1+self.idx,self.scm,self.ty, self.tidx, self.cm, str(self.bb), str(self.pa))   
    def __str__(self):
        return self.label


class Prim(object):
    def __init__(self, item, fd):
        self.fd = fd 
        self.numNode = item[0,0].view(np.int32)
        self.nodeOffset = item[0,1].view(np.int32)
        self.bb = BB(item[2:4].ravel()[:6])

        self.midx = item[1,1].view(np.uint32)
        self.ridx = item[1,2].view(np.uint32)
        self.pidx = item[1,3].view(np.uint32)
        self.mname = self.fd.getName(self.midx)

        levelorder = list(map(lambda item:Node(item, fd), self.fd.node[self.nodeOffset:self.nodeOffset+self.numNode]))
        self.node = levelorder

    def __repr__(self):
        return "Prim %3d %5d %30s : %s" % (self.numNode, self.nodeOffset, str(self.bb), self.mname)   

    def __str__(self):
        return "\n".join([repr(self)]+list(map(repr, self.node))+list(map(lambda n:repr(n.tran), self.node)))

    def __getitem__(self, nodeIdx):
        return self.node[nodeIdx]

class Solid(object):
    @classmethod
    def DecodeLabel(cls, item):
        return item.tobytes()[:16].split(b'\0')[0].decode("utf-8")     # TODO: test full 8 char label  

    def __init__(self, item, fd):
        self.item = item  
        self.fd = fd 
        self.label = self.DecodeLabel(item)
        self.numPrim = item[1,0].view(np.int32)
        self.primOffset = item[1,1].view(np.int32)
        self.prim = list(map(lambda item:Prim(item, fd), self.fd.prim[self.primOffset:self.primOffset+self.numPrim]))
        self.ce = item[2].view(np.float32)

    def __repr__(self):
        return "Solid %10s : %4d %5d  ce %35s  "  % (self.label, self.numPrim, self.primOffset, str(self.ce))

    def __str__(self):
        return "\n".join([repr(self)]+list(map(repr, self.prim)))

    def __getitem__(self, primIdx):
        return self.prim[primIdx]


class Foundry(object):
    BASE = "$TMP/CSG_GGeo/CSGFoundry"
    NAMES = "solid prim node tran itra inst".split()
    def __init__(self, base=None):
        if base is None: base = self.BASE  
        base = os.path.expandvars(base) 
        for n in self.NAMES+["plan"]:
            path = os.path.join(base, "%s.npy" % n)
            if not os.path.exists(path): continue
            setattr( self, n, np.load(path))
        pass   
        self.name = np.loadtxt(os.path.join(base, "name.txt"), dtype=np.object)
        self.label = list(map(Solid.DecodeLabel, self.solid))
        self.solids = list(map(lambda item:Solid(item,self), self.solid))

    def __repr__(self):
        return "\n".join(["Foundry"] + list(map(lambda n:"%10s : %s " % (n,repr(getattr(self, n).shape)), self.NAMES))) 

    def __str__(self):
        return "\n".join(["Foundry"]+ list(map(repr, self.solids)))

    def getName(self, midx):
        return self.name[midx]

    def index(self, solid_label):
        return self.label.index(solid_label) 

    def gt(self):
        return self.node.view(np.int32)[:,3,3] & 0x7fffffff  
    def cm(self):
        return (self.node.view(np.int32)[:,3,3] & 0x80000000) >> 31     # complemented node


    def midx(self):
        return self.prim.view(np.uint32)[:,1,1]  
    def ridx(self):
        return self.prim.view(np.uint32)[:,1,2]  
    def pidx(self):
        return self.prim.view(np.uint32)[:,1,3]  




    def __getitem__(self, arg):
        """
        Access solids nodes prims via string specification::

             fd["d1"]        # solid with label "d1"
             fd["d1/0"]      # 1st prim
             fd["d1/0/0"]    # 1st node of 1st prim

        """
        ret = None
        elem = arg.split("/")

        solid_label = elem[0] if len(elem) > 0 else None
        primIdx = int(elem[1]) if len(elem) > 1 else None
        nodeIdx = int(elem[2]) if len(elem) > 2 else None

        solidIdx = self.index(solid_label) if not solid_label is None else None

        so = None
        pr = None
        nd = None

        if not solidIdx is None:
            so = self.solids[solidIdx]
        pass  
        if not so is None and not primIdx is None:
            pr = so[primIdx]
        pass
        if not pr is None and not nodeIdx is None:
            nd = pr[nodeIdx]
        pass

        if not nd is None:
            return nd
        elif not pr is None:
            return pr
        elif not so is None:
            return so
        else:
            return None
        pass
        return None



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fd = Foundry("$TMP/CSG_GGeo/CSGFoundry")
    print(repr(fd))
    #print(str(fd))
    args = sys.argv[1:] if len(sys.argv) > 1 else "r8/0".split()

    p = None
    n = None
    s = None

    for arg in args: 
        obj = fd[arg] 
        print(arg)
        print(obj)

        if type(obj) is Prim:
            p = obj
        elif type(obj) is Node:
            n = obj
        elif type(obj) is Solid:
            s = obj
        else:
            pass
        pass
    pass

    if not p is None: 
        #print("\n".join(map(repr,p.node))) 
        t = layout_tree(p.node)
        print("layout_tree")
        print(t)
    pass
       

