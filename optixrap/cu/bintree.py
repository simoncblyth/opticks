#!/usr/bin/env python
"""
Working thru approaches to serialization of CSG tree
"""
import logging
log = logging.getLogger(__name__)
import numpy as np

SPHERE = 1
BOX = 2 

UNION = 100
INTERSECTION = 101
DIFFERENCE = 102

desc = { SPHERE:"SPHERE", BOX:"BOX", UNION:"UNION", INTERSECTION:"INTERSECTION", DIFFERENCE:"DIFFERENCE" }


class Node(object):
    @classmethod
    def count_(cls, node, index=0):
        if not node.is_leaf:
            index = cls.count_( node.left, index + 1 ) 
            index = cls.count_( node.right, index + 1) 
        return index 

    @classmethod
    def dump_(cls, node, index=0):
        print "%4s : %s " % ( index, repr(node) )
        if not node.is_leaf:
            index = cls.dump_( node.left, index + 1 ) 
            index = cls.dump_( node.right, index + 1 ) 
        pass
        return index

    @classmethod
    def serialize_(cls, node, a=None):
        if a is None:
            n = Node.count_(node) + 1
            a = np.zeros( (n,4,4), dtype=np.float32 )


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

    def count(self):
        return Node.count_(self)

    def dump(self):
        return Node.dump_(self)
        



if __name__ == '__main__':


    bms = Node(Node(BOX), Node(SPHERE), DIFFERENCE )
    smb = Node(Node(SPHERE), Node(BOX), DIFFERENCE )
    ubo = Node(bms, smb, UNION )

    

    








