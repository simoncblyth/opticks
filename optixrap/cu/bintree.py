#!/usr/bin/env python
"""
Working thru approaches to serialization of CSG tree
"""
import logging
log = logging.getLogger(__name__)
import numpy as np


BREADTH_FIRST = 1 
DEPTH_FIRST = 2 


SPHERE = 1
BOX = 2 
is_shape = lambda c:c in [SPHERE, BOX]

UNION = 100
INTERSECTION = 101
DIFFERENCE = 102
is_operation = lambda c:c in [UNION,INTERSECTION,DIFFERENCE]


desc = { SPHERE:"SPHERE", BOX:"BOX", UNION:"UNION", INTERSECTION:"INTERSECTION", DIFFERENCE:"DIFFERENCE" }



class Node(object):

    @classmethod
    def rdepth_(cls, node):
        """
        Recursive depth first nodelist
        """
        progs = []
        progs.extend([node])
        if not node.is_leaf:
            progs.extend(cls.rdepth_(node.left))
            progs.extend(cls.rdepth_(node.right))
        pass
        return progs
    rdepth = property(lambda self:Node.rdepth_(self))


    @classmethod
    def nodelist_(cls, node, order=BREADTH_FIRST):
        """
        Serialize binary tree nodes into list 
        """
        nls = []

        q = []
        q.append(node)

        while len(q) > 0:
            if order is BREADTH_FIRST:
                t = q.pop(0)   # fifo queue
            else:
                t = q.pop()    # lifo stack
            pass
            if not t is None:
               nls.append(t)
               if not t.is_leaf:
                   q.append(t.left) 
                   q.append(t.right) 
               pass
        pass
        return nls 
    depth = property(lambda self:Node.nodelist_(self, order=DEPTH_FIRST ))
    breadth = property(lambda self:Node.nodelist_(self, order=BREADTH_FIRST ))


    @classmethod
    def serialize_(cls, root):
        """
        :param root: root node of tree to serialize
        :return aa: serialized array
        """
        nls = root.breadth 
        aa = np.zeros( (len(nls),4,4), dtype=np.int32 )
        for i,n in enumerate(nls):
            aa[i,0,0] = n.left if n.is_leaf else n.operation
        pass
        return aa
    serialize = property(lambda self:Node.serialize_(self))

    @classmethod
    def deserialize_(cls, aa, idx=0 ):
        """
        :param aa: serialized array of csg tree 
        :param idx: item index to revive
        :return node: deserialized node

        Breadth first serialization order makes this much simpler to handle. 

        0
        1   2 
        3 4 5 6
        """
        assert idx < len(aa), ("idx invalid", idx, len(aa) ) 
        c = aa[idx,0,0].view(np.uint32)
        node = None
        if is_operation(c):
            idxLeft = 2*idx + 1
            idxRight = 2*idx + 2
            left = cls.deserialize_(aa, idxLeft)
            right = cls.deserialize_(aa, idxRight)
            node =  cls(left, right, c)
        elif is_shape(c):
            node = cls(c)
        else:
            assert False, "bad code %s " % c
        pass
        return node 

    @classmethod
    def roundtrip_(cls, node):
        """
        :param node:
        :return aa: 

        Serialize and deserialize the node, checking the 
        identical repr before and after 
        """ 
        aa = node.serialize    
        node2 = Node.deserialize_(aa)
        assert repr(node) == repr(node2)
        return aa
    roundtrip = property(lambda self:Node.roundtrip_(self))


    def __init__(self, left, right=None, operation=None):
        self.left = left
        self.right = right
        self.operation = operation

    is_leaf = property(lambda self:self.operation is None and self.right is None and not self.left is None)

    def __repr__(self):
        if self.is_leaf:
            return desc[self.left]
        else:
            return "%s(%s,%s)" % ( desc[self.operation], repr(self.left), repr(self.right) )




if __name__ == '__main__':

    bms = Node(Node(BOX), Node(SPHERE), DIFFERENCE )
    smb = Node(Node(SPHERE), Node(BOX), DIFFERENCE )
    ubo = Node(bms, smb, UNION )

    bms.roundtrip
    smb.roundtrip
    ubo.roundtrip




