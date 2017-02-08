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

DIVIDER = 99  # between shapes and operations

UNION = 100
INTERSECTION = 101
DIFFERENCE = 102
is_operation = lambda c:c in [UNION,INTERSECTION,DIFFERENCE]

CODE_JK = 3,3   # item position of shape/operation code

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
    count = property(lambda self:len(self.breadth))


    @classmethod
    def serialize_(cls, root, aa=None, offset=0):
        """
        :param root: root node of tree to serialize
        :param aa: when not None serialization written into this array
        :param offset: offset at which to write the serialization
        :return aa: serialized array
        """
        nls = root.breadth 
        if aa is None:
            aa = np.zeros( (len(nls),4,4), dtype=np.int32 )

        for idx,n in enumerate(nls):
            leaf = n.is_leaf
            uidx = idx + offset
            aa[uidx,CODE_JK[0],CODE_JK[1]] = n.left if leaf else n.operation
            if leaf and n.param is not None:
                aa[uidx, 0] = n.param
            pass
        pass
        return aa
    serialize = property(lambda self:Node.serialize_(self))

    @classmethod
    def concatenate_(cls, nodes ):
        """
        :param nodes: list of separate trees and/or single nodes
        :return composite, offsets:
        """
        counts = map(lambda node:node.count, nodes)
        offsets = np.cumsum( [0] + counts)[:-1]
        dim = (sum(counts),4,4)
        log.info("concatenate_ nodes %s counts %s offsets %s dim %s " % (len(nodes),repr(counts),repr(offsets),dim))

        composite = np.zeros( dim, dtype=np.int32 )
         
        for i, node in enumerate(nodes):
            count = counts[i]
            offset = offsets[i]
            Node.serialize_( node, composite, offset )
        pass
        return composite, offsets


    @classmethod
    def infer_offsets_(cls, composite ):
        """
        For CSG trees operation codes appear in contiguous blocks followed
        by shape codes, so codes can be split to find the component count
        and offsets of trees.  

        Single shape nodes appear as shape codes with no operation codes. 

        ::
 
            deconcatenate_ oidx array([1, 4, 7, 8, 9])                      oruns              [array([1]), array([4]),    array([7, 8, 9])] 
            deconcatenate_ pidx array([ 0,  2,  3,  5,  6, 10, 11, 12, 13]) pruns [array([0]), array([2, 3]), array([5, 6]), array([10, 11, 12, 13])] 

        Currently only single nodes prior to the first tree operation is handled

        TODO: handle the general case of interleaved single nodes and trees, 
              would need to pair up the operations with shapes to identify the 
              non-tree shapes. Complete binary nature of tree makes this doable.

        HMM: would not want to do this kinda thing in CUDA, so better to 
             keep the offsets in the primBuffer to avoid the complication

        """
        codes = composite[:,CODE_JK[0],CODE_JK[1]].view(np.uint32)
        log.info("infer_offsets_ codes %s  " % (repr(codes)))

        oidx = np.where( codes > DIVIDER )[0]  ## indices of operation nodes
        oruns = np.split(oidx, np.where(np.diff(oidx) != 1)[0]+1 )   ## trees appear as contiguous indices
        log.info("infer_offsets_ oidx %s oruns %s " % (repr(oidx), repr(oruns)))

        pidx = np.where( codes < DIVIDER )[0]  ## indices of shape nodes
        pruns = np.split(pidx, np.where(np.diff(pidx) != 1)[0]+1 ) 
        log.info("infer_offsets_ pidx %s pruns %s " % (repr(pidx), repr(pruns)))
 
        offsets = []
        # collect offsets of single nodes with indices 
        # prior to the index of the first operation
        for prun in pruns:
            for pidx in prun:
                if len(oruns) > 0:
                    if pidx < oruns[0][0]:
                        offsets.append(pidx)
                pass

        # collect offsets of first operations in contiguous blocks, ie the starts of trees
        for orun in oruns:
            offsets.append(orun[0])
        pass
        return offsets
 
    @classmethod
    def deconcatenate_(cls, composite, offsets=None ):
        """
        :param composite: array potentially containing multiple serialized trees and/or single nodes
        :param offsets: list of offset indices point at the starts of trees and/or single nodes
        :return nodes: list of deserialized nodes 

        Simple composites can be deconstructed without the offsets parameters
        by infering the offsets using the characteristics of the operation/shape codes
        present in the breadth first serialized csg trees.
        """
        if offsets is None:
            offsets = cls.infer_offsets_(composite) 

        log.info("deconcatenate_ offsets %s " % (repr(offsets)))
        nodes = []
        for offset in offsets:
            node = Node.deserialize_(composite, idx=0, offset=offset )
            nodes.append(node)
        pass
        return nodes 


    @classmethod
    def deserialize_(cls, aa, idx=0, offset=0 ):
        """
        :param aa: serialized array of csg tree 
        :param idx: item index to revive
        :param offset: item offset within the array to the start of the tree at idx=0
        :return node: deserialized node

        Breadth first serialization order makes this much simpler to handle. 

        The offset is intended to allow composites such as separate trees 
        within one array OR just a simple box container holding the tree.

        0
        1   2 
        3 4 5 6
        """
        uidx = offset + idx

        assert uidx < len(aa), ("uidx/idx/offset/len(aa) INVALID INDEX", uidx,idx,offset, len(aa) ) 
        c = aa[uidx,CODE_JK[0],CODE_JK[1]].view(np.uint32)

        node = None
        if is_operation(c):
            idxLeft = 2*idx + 1
            idxRight = 2*idx + 2
            left = cls.deserialize_(aa, idxLeft, offset=offset)
            right = cls.deserialize_(aa, idxRight, offset=offset)
            node =  cls(left, right, c)
        elif is_shape(c):
            node = cls(c)
        else:
            assert False, "bad code %s " % c
        pass
        #log.info("deserialize_ idx %s offset %s uidx %s node %s " % (idx, offset, uidx, repr(node)))
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


    def __init__(self, left, right=None, operation=None, param=None):
        self.left = left
        self.right = right
        self.operation = operation
        self.param = param

    is_leaf = property(lambda self:self.operation is None and self.right is None and not self.left is None)

    def __repr__(self):
        if self.is_leaf:
            return desc[self.left]
        else:
            return "%s(%s,%s)" % ( desc[self.operation], repr(self.left), repr(self.right) )




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    box = Node(BOX, param=[0,0,0,1000])

    bms = Node(Node(BOX, param=[0,0,-100,200]),  Node(SPHERE,param=[0,0,-100,200]), DIFFERENCE )
    smb = Node(Node(SPHERE,param=[0,0,100,300]), Node(BOX,param=[0,0,100,300]), DIFFERENCE )
    ubo = Node(bms, smb, UNION )

    bms.roundtrip
    smb.roundtrip

    ubo_aa = ubo.roundtrip



    nodes = [box, box, box, bms, smb, box, ubo, box] 

    comp_aa, offsets = Node.concatenate_( nodes )
    nodes2 = Node.deconcatenate_( comp_aa, offsets )   

    assert repr(nodes) == repr(nodes2)




