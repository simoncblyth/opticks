#!/usr/bin/env python
"""
        # hmm: diddling with argument objects setting parent pointers caused difficult to find bugs 
        # and forced use of cloning as workaround... instead try to live without
        #
        #if not operation is None:
        #    left.parent = self 
        #    if not right is None:
        #        right.parent = self 
        #pass

    #def clone(self):
    #    if self.is_operation:
    #        cleft = self.left.clone() 
    #        cright = self.right.clone() 
    #    else:
    #        cleft = None
    #        cright = None
    #    pass
    #    return Node(shape=self.shape, left=cleft, right=cright, operation=self.operation, param=self.param, name=self.name)

"""
import logging
log = logging.getLogger(__name__)
import numpy as np

EMPTY = 0 
SPHERE = 1
BOX = 2 
is_shape = lambda c:c in [EMPTY,SPHERE, BOX]

DIVIDER = 99  # between shapes and operations

UNION = 100
INTERSECTION = 101
DIFFERENCE = 102
is_operation = lambda c:c in [UNION,INTERSECTION,DIFFERENCE]


desc = { EMPTY:"EM", SPHERE:"SP", BOX:"BX", UNION:"U", INTERSECTION:"I", DIFFERENCE:"D" }


class Node(object):
    def __init__(self, idx=None, l=None, r=None, **kwa):
        """
        :param idx: 1-based levelorder (aka breadth first) tree index, root at 1
        """
        self.idx = idx
        self.l = l
        self.r = r
        self.next_ = None

        # below needed for CSG 
        self.shape = None 
        self.operation = None
        self._param = None
        self.parent = None
        self.name = "unnamed"
        self.apply_(**kwa)

    def _get_param(self):
        return self._param 
    def _set_param(self, v):
        self._param = np.asarray(v) if v is not None else None 
    param = property(_get_param, _set_param)

    def apply_(self, **kwa):
        for k,v in kwa.items():
            if k == "shape":
                self.shape = v
            elif k == "operation":
                self.operation = v
            elif k == "param":
                self.param = v
            elif k == "name":
                self.name = v 
            else:
                log.warning("ignored Node param %s : %s " % (k,v))
            pass 

    def tree_labelling(self):
        """
        * call this only on the root node
        """
        # tree labelling
        self.maxidx = Node.levelorder_i(self)
        self.maxdepth = Node.depth_r(self)
        Node.postorder_threading_r(self)


    @classmethod
    def dress(cls, root):
        """
        """
        leftop = Node.leftmost(root)
        node = leftop
        while node is not None:
            if node.is_leaf:
                assert 0, "not expecting leaves" 
            elif node.is_bileaf:
                node.apply_(operation=DIFFERENCE)
                pidx = node.idx
                lidx = node.l.idx
                ridx = node.r.idx 
                node.l.apply_(shape=SPHERE, param=[pidx*10,lidx*10,0,100] )             
                node.r.apply_(shape=SPHERE, param=[pidx*10,ridx*10,0,100] )             
            else:
                node.apply_(operation=UNION)
            pass
            log.info(" dress %r " % node )
            node = node.next_ 


    @classmethod
    def traverse(cls, leftop, label="traverse"):
        """
        """
        print "%s : following thread..." % label
        node = leftop
        while node is not None:
            print node
            node = node.next_ 

    def __repr__(self):
        if self.is_bare:
            if self.l is not None and self.r is not None:
                return "Node(%d,l=%r,r=%r)" % (self.idx, self.l, self.r)
            elif self.l is None and self.r is None:
                return "Node(%d)" % (self.idx)
            else:
                assert 0
        else:
            if self.is_primitive:
                return "%s.%s" % (self.tag, desc[self.shape])
            else:
                return "%s.%s(%r,%r)" % ( self.tag, desc[self.operation],self.l, self.r )

    is_primitive = property(lambda self:self.shape is not None)
    is_operation = property(lambda self:self.operation is not None)
    is_bare = property(lambda self:self.operation is None and self.shape is None)

    is_leaf = property(lambda self:self.l is None and self.r is None)

    # bileaf is an operation applied to two leaf nodes, another name is a triple
    is_bileaf = property(lambda self:not self.is_leaf and self.l.is_leaf and self.r.is_leaf)

    is_left = property(lambda self:self.idx % 2 == 0)
    tag = property(lambda self:"%s%d" % ("p" if self.is_primitive else "o", self.idx))


    @classmethod
    def levelorder_i(cls,root):
        """
        Assign 1-based binary tree levelorder indices, eg for height 3 complete tree::

             1
 
             2            3

             4      5     6       7
 
             8   9  10 11 12  13  14   15

        """
        q = []
        q.append(root)

        idx = 1 
        while len(q) > 0:
           node = q.pop(0)   # bottom of q (ie fifo)

           if node.idx is None:
               node.idx = idx
           else:
               assert node.idx == idx
           pass

           idx += 1

           if not node.l is None:q.append(node.l)
           if not node.r is None:q.append(node.r)
        pass
        return idx - 1



    @classmethod
    def depth_r(cls, node, depth=0):
        """
        Marking up the tree with depth, can be done CPU side 
        during conversion : so recursive is fine
        """
        if node is None:
            return 

        #print node
        node.depth = depth

        maxd = depth
        if node.l is not None: maxd = cls.depth_r(node.l, depth+1)
        if node.r is not None: maxd = cls.depth_r(node.r, depth+1)
    
        return maxd

    @classmethod
    def progeny_i(cls,root):
        """

        1

        2          3

        4    5     6      7

        8 9  10 11 12 13 14 

        """

        nodes = []
        q = []
        q.append(root)

        while len(q) > 0:
           node = q.pop(0)   # bottom of q (ie fifo)
           nodes.append(node)
           if not node.l is None:q.append(node.l)
           if not node.r is None:q.append(node.r)
        pass
        return nodes


    @classmethod
    def label_r(cls, node, idx, label):
        if node.idx == idx:
            setattr(node, label, 1)

        if node.l is not None:cls.label_r(node.l, idx, label)
        if node.r is not None:cls.label_r(node.l, idx, label)

    @classmethod
    def leftmost_leaf(cls, root):
        n = root.l  
        while n is not None:
            n = n.l
        return n
 
    @classmethod
    def leftmost(cls, root):
        """
        :return: leftmost internal or operation node 
        """
        l = root 
        while l.l is not None:
            if l.l.is_leaf:
                break
            else:
                l = l.l
            pass

        #assert not l.is_leaf
        return l

    @classmethod
    def postorder_r(cls, root, nodes=[],leaf=True):
        """ 
        :param root:
        :param nodes: list 
        :param leaf: bool control of inclusion of leaf nodes with internal nodes

        Recursive postorder traversal
        """
        if root.l is not None:
            cls.postorder_r(root.l,nodes, leaf=leaf) 
        if root.r is not None:
            cls.postorder_r(root.r,nodes, leaf=leaf)
 
        if not leaf and root.is_leaf:
            pass
        else: 
            nodes.append(root)

        return nodes


    @classmethod
    def postorder_threading_r(cls, root):
        nodes = cls.postorder_r(root, nodes=[], leaf=False)
        for i in range(len(nodes)):
            node = nodes[i]
            next_ = nodes[i+1] if i < len(nodes)-1 else None
            node.next_ = next_


    @classmethod
    def postOrderIterative(cls,root): 
        """
        # iterative postorder traversal using
        # two stacks : nodes processed 

        ::

              1

             [2]                 3

             [4]     [5]         6     7

             [8] [9] [10] [11]  12 13  14 15

        ::

            In [25]: Node.postOrderIterative(root3.l)
            Out[25]: 
            [Node(8),
             Node(9),
             Node(4,l=Node(8),r=Node(9)),
             Node(10),
             Node(11),
             Node(5,l=Node(10),r=Node(11)),
             Node(2,l=Node(4,l=Node(8),r=Node(9)),r=Node(5,l=Node(10),r=Node(11)))]

            In [26]: Node.postOrderIterative(root3.l.l)
            Out[26]: [Node(8), Node(9), Node(4,l=Node(8),r=Node(9))]

        """ 
        if root is None:
            return         
         
        nodes = []
        s = []
         
        nodes.append(root)
         
        while len(nodes) > 0:
             
            node = nodes.pop()
            s.append(node)
         
            if node.l is not None:
                nodes.append(node.l)
            if node.r is not None :
                nodes.append(node.r)
     
        #while len(s) > 0:
        #    node = s.pop()
        #    print node.d,
     
        return list(reversed(s))






root0 = Node(1)
root0.name = "root0"

root1 = Node(1, l=Node(2), r=Node(3))
root1.name = "root1"

root2 = Node(1, 
                l=Node(2, 
                          l=Node(4), 
                          r=Node(5)
                      ), 
                r=Node(3, l=Node(6), 
                          r=Node(7)
                      ) 
            )
root2.name = "root2"


root3 = Node(1, 
                l=Node(2, 
                          l=Node(4,
                                    l=Node(8),
                                    r=Node(9)
                                ), 
                          r=Node(5,
                                    l=Node(10),
                                    r=Node(11),
                                )
                      ), 
                r=Node(3, l=Node(6,
                                    l=Node(12),
                                    r=Node(13),
                                ), 
                          r=Node(7,
                                    l=Node(14),
                                    r=Node(15)
                                )
                      ) 
            )
root3.name = "root3"


root4 = Node(1, 
                l=Node(2, 
                          l=Node(4,
                                    l=Node(8, 
                                              l=Node(16),
                                              r=Node(17)
                                          ),
                                    r=Node(9,
                                              l=Node(18),
                                              r=Node(19)
                                          )
                                ), 
                          r=Node(5,
                                    l=Node(10,
                                              l=Node(20),
                                              r=Node(21)
                                          ),
                                    r=Node(11,
                                              l=Node(22),
                                              r=Node(23)
                                          ),
                                )
                      ), 
                r=Node(3, l=Node(6,
                                    l=Node(12,
                                              l=Node(24),
                                              r=Node(25)
                                          ),
                                    r=Node(13,
                                              l=Node(26),
                                              r=Node(27)
                                          ),
                                ), 
                          r=Node(7,
                                    l=Node(14,
                                              l=Node(28),
                                              r=Node(29)
                                          ),
                                    r=Node(15,
                                              l=Node(30),
                                              r=Node(31)
                                          )
                                )
                      ) 
            )
root4.name = "root4"


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    for tree in [root0, root1, root2, root3, root4]:
        print "%20s : %s " % (tree.name, tree)

        nodes = Node.postorder_r(tree,nodes=[], leaf=False)
        print "Node.postorder_r(%s, leaf=False)\n" % tree.name + "\n".join(map(repr,nodes)) 

        tree.tree_labelling() 

        lpr = Node.leftmost_leaf(tree)
        lop = Node.leftmost(tree)

        print "Node.leftmost(%s) : %s " % (tree.name, lpr )
        print "Node.leftmost_nonleaf(%s) : %s " % (tree.name, lop )

        Node.traverse(lpr, "left prim")
        Node.traverse(lop, "left operation")

        Node.dress(tree)




