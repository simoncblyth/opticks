#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)

TREE_NODES = lambda height:( (0x1 << (1+(height))) - 1 )

class CSG(object):

    SPHERE = 1
    BOX = 2

    UNION = 100
    INTERSECTION = 101
    DIFFERENCE = 102

    @classmethod
    def enum(cls):
        return filter(lambda kv:type(kv[1]) is int,cls.__dict__.items())

    @classmethod
    def desc(cls, typ):
        kvs = filter(lambda kv:kv[1] is typ, cls.enum())
        return kvs[0][0] if len(kvs) == 1 else "UNKNOWN"

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
        return "%s(%s)" % (self.desc(self.typ), ",".join(map(repr,filter(None,[self.left,self.right]))))

    def union(self, other):
        return CSG(typ=CSG.UNION, left=self, right=other)

    def subtract(self, other):
        return CSG(typ=CSG.DIFFERENCE, left=self, right=other)

    def intersect(self, other):
        return CSG(typ=CSG.INTERSECTION, left=self, right=other)

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


