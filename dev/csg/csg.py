#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

# bring in enum values from sysrap/OpticksCSG.h
from opticks.sysrap.OpticksCSG import OpticksCSG 

TREE_NODES = lambda height:( (0x1 << (1+(height))) - 1 )

class CSG(OpticksCSG):

    @classmethod
    def depth_r(cls, node, depth=0):
        assert node is not None
        node.depth = depth
        if node.left is None and node.right is None:
            ldepth = rdepth = depth
        else:
            ldepth = cls.depth_r(node.left, depth+1)
            rdepth = cls.depth_r(node.right, depth+1)
        pass
        return max(ldepth, rdepth)

    @classmethod
    def analyse(cls, root):
        root.maxdepth = cls.depth_r(root, 0)
        root.totnodes = TREE_NODES(root.maxdepth)
        log.info("%s %s " % (repr(root), cls.rootrepr(root)))

    @classmethod
    def rootrepr(cls, root):
        assert hasattr(root, 'maxdepth')
        return "maxdepth:%d totnodes:%d " % (root.maxdepth, root.totnodes)

    def __init__(self, typ, left=None, right=None, parent=None):
        self.typ = typ
        self.left = left
        self.right = right
        self.parent = parent

    def __repr__(self):
        return "%s(%s)" % (OpticksCSG.desc(self.typ), ",".join(map(repr,filter(None,[self.left,self.right]))))

    def union(self, other):
        return CSG(typ=self.UNION, left=self, right=other)

    def subtract(self, other):
        return CSG(typ=self.DIFFERENCE, left=self, right=other)

    def intersect(self, other):
        return CSG(typ=self.INTERSECTION, left=self, right=other)

    def __add__(self, other):
        return self.union(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __mul__(self, other):
        return self.intersect(other)


if __name__ == '__main__':
    pass
    s = CSG(CSG.SPHERE)
    b = CSG(CSG.BOX)

    smb = s - b 

    for i in range(5):
        smb *= b 
        CSG.analyse(smb)    


