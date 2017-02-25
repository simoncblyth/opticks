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
import logging, copy
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
    def __init__(self, idx=0, l=None, r=None, **kwa):
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
        * hmm should make tree complete first to get expected levelorder indices
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
                #assert 0, "not expecting leaves" 
                node.apply_(shape=SPHERE, param=[0,0,0,100])
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

    is_left = property(lambda self:self.idx % 2 == 0) # convention using 1-based levelorder index
    tag = property(lambda self:"%s%d" % ("p" if self.is_leaf else "o", self.idx))

    @classmethod
    def nodecount_r(cls, root):
        if root is None:
            return 0
        return 1 + cls.nodecount_r(root.l) + cls.nodecount_r(root.r)

    @classmethod
    def is_complete_r(cls, root, index=1, nodecount=None, debug=False):
        if nodecount is None:
            nodecount = cls.nodecount_r(root)

        if index >= nodecount: 
            if debug:
               print "index %d >= nodecount %d -> not complete for node %r" % (index, nodecount, root)
            return False

        return cls.is_complete_r(root.l, index=2*index, nodecount=nodecount, debug=debug ) and cls.is_complete_r(root.r, index=2*index+1, nodecount=nodecount, debug=debug)


    @classmethod
    def lheight(cls, root):
        level = 0
        p = root
        while p is not None:
           p = p.l
           level += 1
        pass
        return level


    @classmethod
    def is_perfect_i(cls, root):
        """
        My definition of a perfect binary tree:

        * all leaves are at the same depth, same as maxdepth
        * all non-leaves have non None left and right children

        """
        maxdepth = cls.depth_r(root)  # label nodes with depth, 0 at root
        leafdepth = None
        q = []
        q.append(root)
        while len(q) > 0:
            node = q.pop(0)  # fifo
            if node.is_leaf: 
                if leafdepth is None:
                    leafdepth = node.depth
                    if leafdepth != maxdepth:
                        return False
                    pass
                pass
                if node.depth != leafdepth:
                    return False
                pass
            else:
                if node.l is None or node.r is None:
                    return False
                pass
            pass
            if node.l is not None:q.append(node.l)
            if node.r is not None:q.append(node.r)
        pass
        return True
 
    @classmethod
    def make_perfect_i(cls, root):
        while not cls.is_perfect_i(root): 
            maxdepth = cls.depth_r(root)  # label nodes with depth, 0 at root
            q = []
            q.append(root)
            while len(q) > 0:
                node = q.pop(0)   # bottom of q (ie fifo)
                assert node.depth <= maxdepth

                if node.is_leaf and node.depth < maxdepth:
                    l = Node()
                    r = Node()
                    l.depth = node.depth+1
                    r.depth = node.depth+1
                    node.l = l
                    node.r = r
                pass
                if node.l is not None:q.append(node.l)
                if node.r is not None:q.append(node.r)
            pass
        pass

    @classmethod
    def is_almost_complete_i(cls, root):
        """
        levelorder traverse of complete binary tree should see all the 
        leaves together at the last level. 

        Hmm consider pruing 6,7 in the below so 3 becomes a leaf...
        then all leaves are together 3,4,5
        but 3 is at higher level ?

        ::

             1
 
             2            3

             4      5     6       7

        """

        if root is None:return True
        q = []
        q.append(root)

        leaves = False
    
        idx = 1 
        while len(q) > 0:
           node = q.pop(0)   # bottom of q (ie fifo)

           if node.l is not None:
               if leaves:return False
               q.append(node.l)
           else:
               leaves = True
           pass

           if node.r is not None:
               if leaves:return False
               q.append(node.r)
           else:
               leaves = True
           pass
        pass
        return True




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

           if node.idx is 0:
               node.idx = idx
           else:
               assert node.idx == idx
           pass

           idx += 1

           if node.l is not None:q.append(node.l)
           if node.r is not None:q.append(node.r)
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

        ldepth = depth
        rdepth = depth

        if node.l is not None: ldepth = cls.depth_r(node.l, depth+1)
        if node.r is not None: rdepth = cls.depth_r(node.r, depth+1)
    
        return max(ldepth, rdepth)
        

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
            log.info("label_r setting label %s idx %d node %r  " % (label, idx, node))
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
    def parenting_r(cls, root, parent=None):
        if root.l is not None:
            cls.parenting_r(root.l, parent=root)
        pass
        if root.r is not None:
            cls.parenting_r(root.r, parent=root)
        pass
        root.parent = parent 


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









cbox = Node(shape=BOX, param=[0,0,0,100], name="cbox")
lbox = Node(shape=BOX, param=[-200,0,0,50], name="lbox")
rbox = Node(shape=BOX, param=[ 200,0,0,50], name="rbox")

lrbox = Node(operation=UNION, l=lbox,  r=rbox, name="lrbox") 

bms = Node(name="bms",operation=DIFFERENCE, l=Node(shape=BOX, param=[0,0,0,200], name="box"),  r=Node(shape=SPHERE,param=[0,0,0,150],name="sph"))
smb = Node(name="smb",operation=DIFFERENCE, l=Node(shape=SPHERE,param=[0,0,0,200], name="sph"), r=Node(shape=BOX,param=[0,0,0,150], name="box"))
ubo = Node(name="ubo",operation=UNION, l=bms, r=lrbox )
bmslrbox = Node(name="bmslrbox", operation=UNION, l=Node(name="bmsrbox", operation=UNION, l=bms, r=rbox),r=lbox ) 

bmsrbox = Node(name="bmsrbox", operation=UNION, l=bms, r=rbox )
bmslbox = Node(name="bmslbox", operation=UNION, l=bms, r=lbox )
smblbox = Node(name="smblbox", operation=UNION, l=smb, r=lbox )


# bmslrbox : 
#         U( bms_rbox_u : 
#                U( bms : 
#                         D(bms_box : BX ,
#                           bms_sph : SP ),
#                      rbox : BX ),
#                  lbox : BX ) 
#




bmsrlbox = Node( name="bmsrlbox", operation=UNION, l=bmslbox, r=rbox ) 

csph = Node(shape=SPHERE, param=[0,0,0,100], name="csph")
lsph = Node(shape=SPHERE, param=[-50,0,0,100], name="lsph")
rsph = Node(shape=SPHERE, param=[50,0,0,100], name="rsph")

lrsph_u = Node(operation=UNION, l=lsph, r=rsph, name="lrsph_u")
lrsph_i = Node(operation=INTERSECTION, l=lsph, r=rsph, name="lrsph_i")
lrsph_d = Node(operation=DIFFERENCE, l=lsph, r=rsph, name="lrsph_d")


def test_is_complete():
    """
    ::

         Node(1,l=Node(2,l=Node(4),r=Node(5)),r=Node(3,l=Node(6),r=Node(7)))
         Node(1,l=Node(2,l=Node(4),r=Node(5)),r=Node(3))

    """
    root2c = copy.deepcopy(root2)    
    assert Node.is_complete_i(root2c) 
    assert root2c.r.is_bileaf and root2c.r.idx == 3
    assert root2c.r.l.is_leaf and root2c.r.l.idx == 6
    assert root2c.r.r.is_leaf and root2c.r.r.idx == 7

    # prune two right most leaves 
    root2c.r.l = None
    root2c.r.r = None         
    assert Node.is_complete_i(root2c) == True
    assert Node.is_complete_r(root2c) == False   # different defn 


def test_make_perfect():
    """
    """
    root2c = copy.deepcopy(root2)    
    assert Node.is_perfect_i(root2) == True
    assert Node.is_perfect_i(root2c) == True

    # prune two right most leaves 
    root2c.r.l = None
    root2c.r.r = None         

    assert Node.is_perfect_i(root2c) == False

    print "root2: %s " % root2
    print "root2c: %s (pruned)" % root2c

    Node.make_perfect_i(root2c)

    print "root2c: %s (restored)" % root2c

    assert Node.is_perfect_i(root2c) == True



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    #test_is_complete()
    test_make_perfect()



if 0:
    trees = [bmsrlbox]

    for tree in trees:
        Node.levelorder_i(tree)
        print "%20s : %s " % (tree.name, tree)
        Node.complete_and_full_r(tree)
        

      


if 0:
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




